import os
import io
import json
import numpy as np
import cv2
import streamlit as st
import torch
import folium
import re
from typing import Optional
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

# AI 모델 관련 라이브러리
from paddleocr import PaddleOCR
from sqlmodel import Field, Session, SQLModel, create_engine, select
from transformers import (
    AutoProcessor, AutoModelForImageClassification, 
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DetrImageProcessor, DetrForObjectDetection
)
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi

# 환경 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ---------------------------------------------------------
# 1. DB 모델 및 초기화
# ---------------------------------------------------------
class Document(SQLModel, table=True):
    __table_args__ = {"extend_existing": True} 
    
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    doc_type: str 
    content: str 
    summary: str
    keywords: str
    structured_data: str 
    upload_date: datetime = Field(default_factory=datetime.now)
    image_data: bytes
    embedding: Optional[str] = None

engine = create_engine("sqlite:///archive.db")
SQLModel.metadata.create_all(engine)
kiwi = Kiwi()

# ---------------------------------------------------------
# 2. AI 모델 로딩 (캐싱)
# ---------------------------------------------------------
@st.cache_resource
def load_all_models():
    # PaddleOCR 기본 설정 (안전한 파라미터만 사용)
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='korean'
    )
    dit_p = AutoProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    dit_m = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
    obj_p = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    obj_m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    sum_t = AutoTokenizer.from_pretrained("gogamza/kobart-summarization")
    sum_m = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-summarization")
    emb_m = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return (dit_p, dit_m, ocr, obj_p, obj_m, sum_t, sum_m, emb_m)

# ---------------------------------------------------------
# 3. 보조 분석 함수
# ---------------------------------------------------------
def get_text_from_ocr(ocr_result):
    """PaddleOCR 결과 리스트에서 텍스트만 추출하는 안전한 함수"""
    try:
        if not ocr_result:
            print("[DEBUG] OCR 결과가 None 또는 빈 값입니다.")
            return ""
        if not ocr_result[0]:
            print("[DEBUG] OCR 결과[0]이 None입니다.")
            return ""
        
        text_list = []
        for idx, line in enumerate(ocr_result[0]):
            if line and len(line) >= 2 and line[1]:
                text_list.append(line[1][0])
        
        result = " ".join(text_list)
        print(f"[DEBUG] OCR 추출 완료: {len(result)}글자, 라인 수: {len(text_list)}")
        return result
    except Exception as e:
        print(f"[DEBUG] OCR 파싱 에러: {str(e)}")
        return ""

def extract_photo_metadata(image):
    metadata = {
        'width': image.width, 'height': image.height,
        'camera_model': '정보 없음', 'taken_date': '정보 없음', 
        'location_address': '정보 없음', 'lat': None, 'lng': None
    }
    try:
        exif_data = image._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "Model": metadata['camera_model'] = str(value).strip()
                elif tag in ["DateTime", "DateTimeOriginal"]: 
                    metadata['taken_date'] = str(value).replace(':', '-', 2)
                elif tag == "GPSInfo" and isinstance(value, dict):
                    gps_data = {GPSTAGS.get(t, t): value[t] for t in value}
                    if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                        def to_decimal(dms, ref):
                            d, m, s = [float(x) for x in dms]
                            res = d + m/60.0 + s/3600.0
                            return -res if ref in ['S', 'W'] else res
                        metadata['lat'] = to_decimal(gps_data['GPSLatitude'], gps_data['GPSLatitudeRef'])
                        metadata['lng'] = to_decimal(gps_data['GPSLongitude'], gps_data['GPSLongitudeRef'])
                        try:
                            geolocator = Nominatim(user_agent="geo_archive_v4")
                            loc = geolocator.reverse(f"{metadata['lat']}, {metadata['lng']}", language='ko')
                            if loc: metadata['location_address'] = loc.address
                        except: pass
    except: pass
    return metadata

def detect_photo_objects(image, processor, model):
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        objs = [model.config.id2label[label.item()] for label in results["labels"]]
        return list(set(objs))
    except: return []

def generate_photo_keywords(metadata, objects):
    kws = ["사진"] + objects
    if metadata['camera_model'] != '정보 없음': kws.append(metadata['camera_model'])
    if metadata['location_address'] != '정보 없음':
        kws.extend([x.strip() for x in metadata['location_address'].split(',')[:2]])
    return ", ".join(list(dict.fromkeys(kws)))

# ---------------------------------------------------------
# 4. 메인 프로세싱 함수 (통합 분석)
# ---------------------------------------------------------
def process_document(uploaded_file, models):
    (dit_p, dit_m, ocr, obj_p, obj_m, sum_t, sum_m, emb_m) = models
    file_bytes = uploaded_file.read()
    raw_img = Image.open(io.BytesIO(file_bytes))
    orig_img = raw_img.convert("RGB")
    
    # 1. 문서 분류 (원본 사이즈 기반)
    inputs = dit_p(images=orig_img, return_tensors="pt")
    label = dit_m.config.id2label[dit_m(**inputs).logits.argmax(-1).item()].lower()
    
    # 기초 이미지 변환 (OpenCV 포맷)
    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    
    # 1차 OCR (분류 보조용)
    print(f"[DEBUG] 이미지 크기: {img_cv.shape}")
    ocr_res_init = ocr.ocr(img_cv)
    print(f"[DEBUG] 초기 OCR 결과 타입: {type(ocr_res_init)}")
    initial_text = get_text_from_ocr(ocr_res_init)
    print(f"[DEBUG] 초기 텍스트: '{initial_text[:100] if initial_text else '(없음)'}'...")

    # 문서/사진 판별 (개선된 로직)
    doc_keywords = ['receipt', 'invoice', 'form', 'letter', 'advertisement', 'resume', 'news', 'scientific', 'publication', 'memo']
    is_doc = any(x in label for x in doc_keywords) or len(initial_text) > 30
    
    # 추가 판별: 텍스트 밀도 계산 (신문, 문서는 텍스트가 많음)
    if not is_doc and len(initial_text) > 15:
        # 이미지 면적 대비 텍스트 길이 비율로 판단
        img_area = img_cv.shape[0] * img_cv.shape[1]
        text_density = len(initial_text) / (img_area / 10000)  # 만 픽셀당 문자 수
        if text_density > 2.0:  # 문자 밀도가 높으면 문서로 판정
            is_doc = True
    
    if is_doc:
        doc_type = "Document"
        # --- [적극적인 해상도 향상] ---
        height, width = img_cv.shape[:2]
        
        print(f"[DEBUG] 원본 이미지 크기: {width}x{height}")
        
        # 해상도가 너무 낮으면 대폭 확대 (최소 1500px 보장)
        target_width = 1500
        if width < target_width:
            scale = target_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            print(f"[DEBUG] 이미지 확대: {scale:.2f}배 -> {new_width}x{new_height}")
            img_cv_enlarged = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            img_cv_enlarged = img_cv.copy()
        
        # 확대된 이미지로 OCR
        print(f"[DEBUG] OCR 시작 (확대된 이미지)")
        ocr_res_enlarged = ocr.ocr(img_cv_enlarged)
        text_enlarged = get_text_from_ocr(ocr_res_enlarged)
        
        # 추가 시도: 샤프닝 적용
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img_cv_enlarged, -1, kernel_sharpen)
        ocr_res_sharp = ocr.ocr(sharpened)
        text_sharp = get_text_from_ocr(ocr_res_sharp)
        
        # 가장 긴 결과 선택
        results = [
            (text_enlarged, "확대"),
            (text_sharp, "확대+샤프닝"),
            (initial_text, "원본")
        ]
        full_text_raw, best_method = max(results, key=lambda x: len(x[0]))
        print(f"[DEBUG] 최적 방법: {best_method}, 텍스트 길이: {len(full_text_raw)}")
        
        # UI 표시용 이미지
        processed_gray = cv2.cvtColor(img_cv_enlarged, cv2.COLOR_BGR2GRAY)
        input_for_ocr = img_cv_enlarged
        
        # 텍스트 정제 (노이즈 문자 제거)
        cleaned_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', full_text_raw)  # 단일 알파벳만 제거
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        full_text = cleaned_text if len(cleaned_text) > 10 else full_text_raw
        # --- [전처리 로직 끝] ---

        # 키워드 추출
        kw_list = [t.form for t in kiwi.tokenize(full_text) if t.tag in ['NNG', 'NNP']]
        final_keywords = ", ".join(list(dict.fromkeys(kw_list))[:10])
        
        # 요약 생성
        try:
            if len(full_text) > 20:
                s_inputs = sum_t([full_text], max_length=512, return_tensors="pt", truncation=True)
                s_ids = sum_m.generate(
                    s_inputs["input_ids"], 
                    num_beams=4,
                    max_length=128,
                    min_length=10,
                    repetition_penalty=3.5,
                    no_repeat_ngram_size=2,
                    eos_token_id=sum_t.eos_token_id
                )
                final_summary = sum_t.decode(s_ids[0], skip_special_tokens=True).strip()
            else:
                final_summary = "요약할 텍스트가 부족합니다."
        except: 
            final_summary = f"{full_text[:30]}... 내용의 문서"
            
        # UI 출력용 이미지 (RGB 변환)
        processed_img_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
        structured_data = {}
        
    else:
        doc_type = "Photo"
        processed_img_rgb = np.array(orig_img)
        full_text = ""
        meta = extract_photo_metadata(raw_img)
        objects = detect_photo_objects(orig_img, obj_p, obj_m)
        final_keywords = generate_photo_keywords(meta, objects)
        final_summary = f"[{meta['taken_date']}] 촬영 사진. 탐지 객체: {', '.join(objects)}"
        structured_data = {'exif': meta, 'objects': objects}

    embedding = emb_m.encode(full_text + " " + final_keywords).tolist()
    return (doc_type, full_text, final_summary, final_keywords, structured_data, file_bytes, embedding, processed_img_rgb)

# ---------------------------------------------------------
# 5. UI 및 지도 표시
# ---------------------------------------------------------
def display_photo_locations(items):
    locs = []
    for d in items:
        try:
            sd = json.loads(d.structured_data)
            if 'exif' in sd and sd['exif'].get('lat') is not None:
                locs.append({
                    'lat': sd['exif']['lat'], 
                    'lng': sd['exif']['lng'], 
                    'name': d.filename, 
                    'addr': sd['exif'].get('location_address', '주소 미상')
                })
        except: continue
    
    if locs:
        m = folium.Map(location=[locs[0]['lat'], locs[0]['lng']], zoom_start=12)
        for l in locs:
            folium.Marker(
                [l['lat'], l['lng']], 
                popup=folium.Popup(f"<b>{l['name']}</b><br>{l['addr']}", max_width=300),
                tooltip=l['name'],
                icon=folium.Icon(color='red', icon='camera')
            ).add_to(m)
        st_folium(m, width=1200, height=600)
    else:
        st.info("📍 지도에 표시할 위치 정보(GPS)가 포함된 사진이 없습니다.")

# 메인 실행부
st.set_page_config(layout="wide", page_title="AI Multi-Archive")
st.title("🌟 멀티모달 AI 통합 아카이브")

models = load_all_models()
t1, t2, t3, t4 = st.tabs(["📤 업로드", "🔍 검색", "📁 아카이브", "📍 지도"])

with t1:
    file = st.file_uploader("이미지 업로드", type=['jpg', 'png', 'jpeg'])
    if file:
        if "res" not in st.session_state or st.session_state.get("fname") != file.name:
            with st.spinner("분석 중..."):
                st.session_state.res = process_document(file, models)
                st.session_state.fname = file.name
        
        r = st.session_state.res
        
        # 디버그용 변수 저장 (세션 상태에 추가 정보 저장)
        if "debug_info" not in st.session_state:
            st.session_state.debug_info = {}
        
        col1, col2 = st.columns(2)
        
        # 원본 이미지 표시
        orig_display = Image.open(io.BytesIO(r[5]))
        col1.image(orig_display, caption="원본", use_container_width=True)
        
        # 전처리 결과 표시 (numpy array를 PIL로 변환)
        if r[0] == "Document":
            col2.image(r[7], caption="전처리 결과", use_container_width=True)
        else:
            col2.image(orig_display, caption="사진 (전처리 없음)", use_container_width=True)
        
        st.write(f"**분류:** {r[0]} | **키워드:** `{r[3]}`")
        
        # OCR 디버그 정보 추가
        with st.expander("🔍 OCR 디버그 정보", expanded=True):
            st.write(f"**추출된 텍스트 길이:** {len(r[1])} 글자")
            
            # 추출된 텍스트 표시
            if r[1]:
                st.text_area("추출된 전체 텍스트", r[1], height=150, key="ocr_text")
                st.caption(f"키워드: {r[3]}")
            else:
                st.text_area("추출된 전체 텍스트", "(텍스트 없음)", height=150, key="ocr_text")
            
            if not r[1] or len(r[1]) < 50:
                st.error("⚠️ OCR 품질 저하: 텍스트를 제대로 추출하지 못했습니다.")
                st.info("""
💡 **문제 진단 체크리스트:**
1. **이미지 해상도**: 최소 1000px 이상 권장 (현재 news.jpg는 해상도가 낮을 수 있음)
2. **글자 크기**: 신문 글씨가 너무 작으면 인식 실패
3. **PaddleOCR 언어팩**: 'korean' 모델이 제대로 다운로드되었는지 확인
4. **이미지 품질**: JPG 압축으로 글자가 흐려졌을 가능성

**해결 방법:**
- Streamlit 재시작 후 캐시 클리어 (좌측 메뉴 > Clear cache)
- 더 고해상도 이미지로 테스트
- PNG 포맷으로 변환 후 재시도
- 터미널 [DEBUG] 메시지 확인
                """)
        
        st.info(f"**요약:** {r[2]}")
        
        if st.button("🚀 최종 저장", type="primary"):
            with Session(engine) as session:
                new_doc = Document(filename=file.name, doc_type=r[0], content=r[1], 
                                   summary=r[2], keywords=r[3], 
                                   structured_data=json.dumps(r[4], ensure_ascii=False),
                                   image_data=r[5], embedding=json.dumps(r[6]))
                session.add(new_doc)
                session.commit()
            st.success("저장 완료!")

with t2:
    q = st.text_input("검색어 (객체, 장소, 내용 등)")
    if q:
        with Session(engine) as session:
            results = session.exec(select(Document).where((Document.content.contains(q)) | (Document.keywords.contains(q)))).all()
            for d in results:
                with st.expander(f"📄 {d.filename} ({d.doc_type})"):
                    sc1, sc2 = st.columns([1, 3])
                    sc1.image(d.image_data)
                    sc2.write(f"**요약:** {d.summary}")
                    sc2.write(f"**키워드:** `{d.keywords}`")

with t3:
    with Session(engine) as session:
        items = session.exec(select(Document).order_by(Document.upload_date.desc())).all()
        for item in items:
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                c1.image(item.image_data)
                c2.write(f"**{item.filename}** ({item.doc_type})")
                c2.caption(f"요약: {item.summary} | 키워드: {item.keywords}")
                if st.button("🗑️ 삭제", key=f"del_{item.id}"):
                    session.delete(item); session.commit(); st.rerun()

with t4:
    st.header("📍 사진 촬영 위치")
    with Session(engine) as session:
        all_docs = session.exec(select(Document)).all()
        display_photo_locations(all_docs)
