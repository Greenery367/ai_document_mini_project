import os
import io
import json
import numpy as np
import cv2
import streamlit as st
import torch
import folium
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
    ocr = PaddleOCR(lang='korean', show_log=False)
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
    
    # 문서 분류 (원본 사이즈)
    inputs = dit_p(images=orig_img, return_tensors="pt")
    label = dit_m.config.id2label[dit_m(**inputs).logits.argmax(-1).item()].lower()
    
    # OCR을 위한 1차 텍스트 추출 (문서 판별용)
    img_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    ocr_res_init = ocr.ocr(img_cv, cls=False)
    initial_text = " ".join([line[1][0] for line in ocr_res_init[0]]) if ocr_res_init and ocr_res_init[0] else ""

    # 문서/사진 판별
    is_doc = any(x in label for x in ['receipt', 'invoice', 'form', 'letter']) or len(initial_text) > 40
    
    if is_doc:
        doc_type = "Document"
        # 문서인 경우 전처리 수행 (확대 + 샤프닝 + 이진화)
        height, width = img_cv.shape[:2]
        img_cv_up = cv2.resize(img_cv, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(img_cv_up, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
        _, processed_img = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 고도화된 전처리 이미지로 다시 OCR
        ocr_res_final = ocr.ocr(img_cv_up, cls=False)
        full_text = " ".join([line[1][0] for line in ocr_res_final[0]]) if ocr_res_final and ocr_res_final[0] else initial_text
        
        kw_list = [t.form for t in kiwi.tokenize(full_text) if t.tag in ['NNG', 'NNP']]
        final_keywords = ", ".join(list(dict.fromkeys(kw_list))[:10])
        
        try:
            s_inputs = sum_t([full_text], max_length=128, return_tensors="pt", truncation=True)
            s_ids = sum_m.generate(s_inputs["input_ids"], num_beams=4, max_length=128)
            final_summary = sum_t.decode(s_ids[0], skip_special_tokens=True).strip()
        except: final_summary = f"{full_text[:30]}... 내용의 문서"
        structured_data = {}
    else:
        doc_type = "Photo"
        # 사진인 경우 전처리 생략 (원본 그대로 유지)
        processed_img = np.array(orig_img)
        full_text = ""
        meta = extract_photo_metadata(raw_img)
        objects = detect_photo_objects(orig_img, obj_p, obj_m)
        final_keywords = generate_photo_keywords(meta, objects)
        final_summary = f"[{meta['taken_date']}] 촬영 사진. 탐지 객체: {', '.join(objects)}"
        structured_data = {'exif': meta, 'objects': objects}

    embedding = emb_m.encode(full_text + " " + final_keywords).tolist()
    return (doc_type, full_text, final_summary, final_keywords, structured_data, file_bytes, embedding, processed_img)

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
        # 첫 번째 위치를 중심으로 지도 생성
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
        col1, col2 = st.columns(2)
        col1.image(r[5], caption="원본")
        col2.image(r[7], caption="전처리 결과 (사진은 원본유지)")
        
        st.write(f"**분류:** {r[0]} | **키워드:** `{r[3]}`")
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