# app.py
"""
FIR PII Extractor (Final Stable Version)
- Extracts FIR metadata from Hindi/English PDFs
- Uses PyMuPDF for text + image rendering (no Poppler)
- EasyOCR for OCR fallback
- Deterministic per-field extractors (regex + fuzzy)
- Streamlit UI with editable fields, evidence, JSON export
"""

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import numpy as np
import easyocr
import re, json, dateparser
from rapidfuzz import process, fuzz
from typing import Dict, Any, List

# -----------------------
# Init resources
# -----------------------
@st.cache_resource(show_spinner=False)
def load_ocr():
    return easyocr.Reader(['en', 'hi'])

reader = load_ocr()

# -----------------------
# OCR helper (via PyMuPDF render)
# -----------------------
def ocr_page(page, dpi=200):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # enhance for OCR
    img = ImageEnhance.Contrast(img).enhance(1.2)
    arr = np.array(img)
    results = reader.readtext(arr, detail=0, paragraph=True)
    return " ".join(results)

# -----------------------
# Extract text from PDF
# -----------------------
def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            text = ocr_page(page)
        texts.append(text)
    return "\n".join(texts)

# -----------------------
# Field extractors
# -----------------------
def extract_fir_no(text: str) -> (str, str):
    m = re.search(r"FIR\s*(No\.?|Number)?\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", text, re.I)
    if m: return m.group(2), m.group(0)
    return "", ""

def extract_date(text: str) -> (str, str):
    m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", text)
    if m:
        d = dateparser.parse(m.group(1), settings={"DATE_ORDER":"DMY"})
        if d: return d.strftime("%Y-%m-%d"), m.group(0)
    return "", ""

def extract_time(text: str) -> (str, str):
    m = re.search(r"(\d{1,2}:\d{2})", text)
    if m: return m.group(1), m.group(0)
    return "", ""

def extract_phone(text: str) -> (str, str):
    m = re.search(r"(\+?\d[\d\s\-]{8,}\d)", text)
    if m:
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 10:
            return "+91" + digits, m.group(0)
        return digits, m.group(0)
    return "", ""

def extract_sections(text: str) -> (List[str], str):
    m = re.findall(r"\b(\d{1,3})\b", text)
    nums = list(dict.fromkeys([n for n in m if int(n) < 600]))
    return nums, ", ".join(nums) if nums else ""

def extract_police_station(text: str) -> (str, str):
    m = re.search(r"(Police Station|P\.S\.|पोलीस ठाणे)[:\-]?\s*([^\n]+)", text, re.I)
    if m: return m.group(2).strip(), m.group(0)
    return "", ""

def extract_district_state(text: str) -> (str, str, str):
    # naive split District (State)
    m = re.search(r"District.*?:\s*([A-Za-z\u0900-\u097F\s]+)", text)
    district, state = "", ""
    if m: district = m.group(1).strip()
    # fuzzy match states
    STATES = ["Maharashtra","छत्तीसगढ़","Uttar Pradesh","Madhya Pradesh","Delhi"]
    best = process.extractOne(text, STATES, scorer=fuzz.WRatio)
    if best and best[1] > 60: state = best[0]
    return district, state, district + " " + state

def extract_address(text: str) -> (str, str):
    m = re.search(r"(Address|पत्ता)[:\-]?\s*([^\n]+)", text)
    if m: return m.group(2).strip(), m.group(0)
    return "", ""

def extract_names(text: str) -> (List[str], str):
    names = []
    for line in text.splitlines():
        if re.search(r"(Name|नाव)", line, re.I):
            parts = re.split(r"[:\-]", line)
            if len(parts) > 1:
                names.append(parts[1].strip())
    return names, "; ".join(names)

# -----------------------
# Main extraction
# -----------------------
def extract_fields(text: str) -> Dict[str, Any]:
    fir_no, ev_fir = extract_fir_no(text)
    date, ev_date = extract_date(text)
    time, ev_time = extract_time(text)
    phone, ev_phone = extract_phone(text)
    sections, ev_sec = extract_sections(text)
    ps, ev_ps = extract_police_station(text)
    dist, state, ev_ds = extract_district_state(text)
    addr, ev_addr = extract_address(text)
    names, ev_names = extract_names(text)

    return {
        "fir_no": fir_no, "date": date, "time": time,
        "phone": phone, "under_sections": sections,
        "police_station": ps, "dist_name": dist, "state_name": state,
        "address": addr, "names": names,
        "_evidence": {
            "fir_no": ev_fir, "date": ev_date, "time": ev_time,
            "phone": ev_phone, "under_sections": ev_sec,
            "police_station": ev_ps, "district_state": ev_ds,
            "address": ev_addr, "names": ev_names
        }
    }

# -----------------------
# Streamlit UI
# -----------------------
st.title("📄 FIR PII Extractor (Final, Offline)")

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    with st.spinner("Extracting..."):
        text = extract_text(pdf_bytes)
        results = extract_fields(text)

    st.subheader("📑 Extracted Text")
    st.text_area("Raw Text", text, height=250)

    st.subheader("🧾 Extracted PII (Editable)")
    edited = {}
    for key, val in results.items():
        if key.startswith("_"): continue
        if isinstance(val, list):
            edited[key] = st.text_area(key, "\n".join(val))
        else:
            edited[key] = st.text_input(key, str(val))

    st.subheader("🔍 Evidence")
    st.json(results["_evidence"])

    if st.button("💾 Download JSON"):
        out = json.dumps(edited, ensure_ascii=False, indent=2)
        st.download_button("Download JSON", out, file_name="fir_pii.json", mime="application/json")
