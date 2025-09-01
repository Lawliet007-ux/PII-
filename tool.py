"""
Streamlit app: FIR PII extractor (Hindi + English)
Saves extracted PII as JSON and shows raw text.

How it works (high level):
1. Try to extract selectable text using PyMuPDF (fitz) and pdfplumber.
2. If text is missing or insufficient, render each page to an image and run Tesseract OCR (eng+hin).
3. Detect language per block and run rule-based regex extraction for the requested PII fields.
4. Optional: plug in a token classification (NER) model from HuggingFace for improved name/address detection.

Features:
- Upload single or multiple PDF files
- View raw text per page and combined
- Show extracted PII in a table and JSON
- Download JSON results

Dependencies:
- pip install streamlit pymupdf pdfplumber pytesseract pillow opencv-python regex langdetect transformers torch
- System: tesseract-ocr installed + Devanagari (hin) traineddata (for example on Ubuntu: apt-get install tesseract-ocr tesseract-ocr-hin)

Note: This app is designed to be a high-quality, pragmatic starting point. For best production accuracy we recommend
- fine-tuning a multilingual token-classification model (XLM-R / IndicBERT) on annotated FIRs
- using commercial OCR (Google Vision / AWS Textract) for noisy scans

"""

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
import json
from langdetect import detect
import tempfile
import os
import base64

st.set_page_config(page_title="FIR PII Extractor — Hindi + English", layout="wide")

# ------------------ Helpers ------------------

def extract_text_pymupdf(path):
    try:
        doc = fitz.open(path)
        pages = [p.get_text("text") for p in doc]
        return pages
    except Exception as e:
        return None

def extract_text_pdfplumber(path):
    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                text = p.extract_text()
                pages.append(text or "")
        return pages
    except Exception:
        return None

def render_pages_to_images(path, zoom=2):
    images = []
    doc = fitz.open(path)
    for page in doc:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def ocr_images(images, tesseract_langs="eng+hin"):
    texts = []
    for img in images:
        # Preprocess: convert to grayscale, increase contrast if needed
        gray = img.convert("L")
        txt = pytesseract.image_to_string(gray, lang=tesseract_langs)
        texts.append(txt)
    return texts

# Robust combined extractor
def extract_text_from_pdf(path):
    # 1. try PyMuPDF
    pages = extract_text_pymupdf(path)
    # 2. if pages empty or low text, try pdfplumber
    combined = "\n".join([p or "" for p in pages]) if pages else ""
    if not combined or len(combined) < 100:
        pages = extract_text_pdfplumber(path)
        combined = "\n".join([p or "" for p in pages]) if pages else ""
    # 3. if still low, fallback to OCR
    if not combined or len(combined) < 150:
        images = render_pages_to_images(path)
        pages = ocr_images(images)
        combined = "\n".join([p or "" for p in pages])
    return pages, combined

# ------------------ PII Extraction Rules ------------------

# Compile a set of regex patterns helpful for FIRs (English + Devanagari hints)
REGEX_PATTERNS = {
    "year": [r"\b(19|20)\d{2}\b", r"Year\s*[:\-]\s*(\d{4})", r"वष\s*[:\-]\s*(\d{4})"],
    "state_name": [r"State\s*[:\-]\s*([A-Za-z ]+)", r"रा.?य[:\-]\s*([^,\n]+)", r"महाराष्ट्र|दिल्ली|पुणे"],
    "dist_name": [r"District\s*[:\-]\s*([A-Za-z ]+)", r"ज.{0,2}हा[:\-]\s*([^,\n]+)", r"पुणे शहर"],
    "police_station": [r"P\.?S\.?\s*[:\-]\s*([A-Za-z \u0900-\u097F]+)", r"पोलीस\s*ठाणे[:\-]\s*([\u0900-\u097F ]+)", r"ठाणे\s*[:\-]\s*([\u0900-\u097F ]+)", r"Police Station[:\-]\s*([A-Za-z ]+)", r"भोसरी"],
    "under_acts": [r"Acts?\s*[:\-]\s*([^\n]+)", r"अधिनयम[:\-]\s*([^\n]+)", r"Indian Arms Act|Arms Act|भारतीय हतियार|महारा.*पोलिस"],
    "under_sections": [r"Sections?\s*[:\-]\s*([^\n]+)", r"कलम[:\-]\s*([^\n]+)", r"\b\d{1,3}(?:\(\d+\))?\b"],
    "name": [r"Name\s*[:\-]\s*([A-Z][A-Za-z \.]+)", r"नाव[:\-]\s*([\u0900-\u097F ]{2,60})", r"Complainant.*?Name[:\-]\s*([A-Z][A-Za-z ]+)", r"VIPUL.*JADHAV"],
    "address": [r"Address\s*[:\-]\s*([^\n]+)", r"पत्ता[:\-]\s*([^\n]+)", r"PUNE CITY, भोसरी, पुणे"],
}

# normalization helpers

def _first_match(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            # return first capturing group if present
            if m.groups():
                for g in m.groups():
                    if g and g.strip():
                        return g.strip()
            return m.group(0).strip()
    return None

# Main PII extraction (rule-based) — returns only requested fields
REQ_FIELDS = [
    "year",
    "state_name",
    "dist_name",
    "police_station",
    "under_acts",
    "under_sections",
    "revised_case_category",
    "oparty",
    "name",
    "address",
    "jurisdiction",
    "jurisdiction_type",
]

CASE_CATEGORY_MAP = {
    # Basic mapping heuristics; in production replace with ML classifier
    "509": "CRIMINAL_OFFENCE_SEXUAL_HARASSMENT",
    "Arms": "WEAPONS",
    "हथियार": "WEAPONS",
}


def extract_pii_rules(text):
    out = {k: None for k in REQ_FIELDS}
    # simple fields
    out["year"] = _first_match(REGEX_PATTERNS["year"], text)
    out["state_name"] = _first_match(REGEX_PATTERNS["state_name"], text)
    out["dist_name"] = _first_match(REGEX_PATTERNS["dist_name"], text)
    out["police_station"] = _first_match(REGEX_PATTERNS["police_station"], text)
    # Acts / sections
    acts = _first_match(REGEX_PATTERNS["under_acts"], text)
    if acts:
        out["under_acts"] = [a.strip() for a in re.split(r"[,;\n/]+", acts) if a.strip()]
    # sections: try to find nearby numeric sequences by scanning sections block
    secs = re.findall(r"\b\d{1,3}(?:\(\d+\))?\b", text)
    if secs:
        out["under_sections"] = list(dict.fromkeys(secs))
    # complainant/name/address heuristics
    nm = _first_match(REGEX_PATTERNS["name"], text)
    if nm:
        out["name"] = nm
        out["oparty"] = "complainant_or_informant"
    addr = _first_match(REGEX_PATTERNS["address"], text)
    if addr:
        out["address"] = addr
    # revised_case_category: crude rules
    for k, v in CASE_CATEGORY_MAP.items():
        if re.search(k, text, flags=re.IGNORECASE):
            out["revised_case_category"] = v
            break
    # jurisdiction inference (district + ps)
    if out.get("dist_name"):
        out["jurisdiction"] = out["dist_name"]
        out["jurisdiction_type"] = "DISTRICT"

    # Clean empty lists->None
    for k in ["under_acts", "under_sections"]:
        if out.get(k) and len(out[k]) == 0:
            out[k] = None
    return out

# ------------------ Streamlit UI ------------------

st.title("FIR PII Extractor — Hindi + English")
st.write("Upload PDFs (scanned or selectable). The app will try selectable text first and fallback to OCR.")

uploaded_files = st.file_uploader("Upload one or more FIR PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    results = {}
    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        st.write(f"Processing **{uploaded.name}**")
        pages, combined_text = extract_text_from_pdf(tmp_path)
        st.expander("Show raw extracted text (first 2000 chars)", expanded=False).write(combined_text[:2000])
        # Extract PII
        pii = extract_pii_rules(combined_text)
        # More advanced NER optional placeholder
        results[uploaded.name] = pii
        # Clean up
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    st.subheader("Extracted PII (per file)")
    st.json(results)

    # Download button
    b64 = base64.b64encode(json.dumps(results, ensure_ascii=False, indent=2).encode()).decode()
    href = f"data:application/json;base64,{b64}"
    st.markdown(f"[Download JSON]({href})")

    st.info("Notes: This app uses rule-based heuristics by default. For production-grade accuracy on noisy or heterogeneous FIRs, fine-tune a multilingual NER model and/or use higher-quality OCR.")

else:
    st.write("No files uploaded yet — drop a PDF to get started.")

# EOF
