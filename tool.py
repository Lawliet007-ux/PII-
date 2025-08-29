import streamlit as st
import fitz
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re
import json

# ---------------------------
# OCR Function
# ---------------------------
def ocr_pdf(file_bytes):
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            pil = page.to_image(resolution=300).original
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_text = pytesseract.image_to_string(th, lang="hin+eng")
            text.append(ocr_text)
    return "\n".join(text)

# ---------------------------
# Main Extraction
# ---------------------------
def extract_text(file_bytes):
    # First try direct text
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = []
    for page in doc:
        t = page.get_text("text")
        if t.strip():
            text.append(t)
    extracted = "\n".join(text)

    # If empty → fallback to OCR
    if not extracted.strip():
        extracted = ocr_pdf(file_bytes)

    return extracted.strip()

# ---------------------------
# Schema Extraction (simple demo)
# ---------------------------
def extract_schema(text):
    schema = {
        "fir_no": None,
        "year": None,
        "state_name": None,
        "dist_name": None,
        "police_station": None,
        "under_acts": [],
        "under_sections": [],
        "revised_case_category": None,
        "oparty": [],
        "jurisdiction": None,
        "jurisdiction_type": None
    }

    m = re.search(r"FIR\s*No[:\- ]?(\d+)", text, re.IGNORECASE)
    if m: schema["fir_no"] = m.group(1)

    m = re.search(r"(\d{4})", text)
    if m: schema["year"] = m.group(1)

    return schema

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📄 FIR / Legal Document Structured PII Extractor (with OCR)")

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"])
if uploaded:
    file_bytes = uploaded.read()
    text = extract_text(file_bytes)

    st.subheader("Extracted Text (Preview)")
    st.text_area("Text", text[:3000], height=300)

    st.subheader("Extracted Structured Schema")
    schema = extract_schema(text)
    st.json(schema)

    st.download_button("⬇️ Download JSON",
                       data=json.dumps(schema, indent=2, ensure_ascii=False),
                       file_name="fir_schema.json",
                       mime="application/json")
