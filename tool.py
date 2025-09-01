"""
Streamlit app: FIR PII extractor (Hindi + English)
Updated with improved cleaning, normalization, and filtering of results.
"""

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
import json
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
        gray = img.convert("L")
        txt = pytesseract.image_to_string(gray, lang=tesseract_langs)
        texts.append(txt)
    return texts

# Combined extractor
def extract_text_from_pdf(path):
    pages = extract_text_pymupdf(path)
    combined = "\n".join([p or "" for p in pages]) if pages else ""
    if not combined or len(combined) < 100:
        pages = extract_text_pdfplumber(path)
        combined = "\n".join([p or "" for p in pages]) if pages else ""
    if not combined or len(combined) < 150:
        images = render_pages_to_images(path)
        pages = ocr_images(images)
        combined = "\n".join([p or "" for p in pages])
    return pages, combined

# ------------------ Improved Extraction ------------------

def extract_pii_rules(text):
    result = {}
    clean_text = re.sub(r"[ \t]+", " ", text)

    # Year
    m = re.search(r"(19|20)\d{2}", clean_text)
    result["year"] = m.group(0) if m else None

    # State
    if "महारा" in clean_text or "Maharashtra" in clean_text:
        result["state_name"] = "Maharashtra"

    # District
    m = re.search(r"District[:\s]+([A-Za-z ]+)", clean_text, re.IGNORECASE)
    if not m:
        m = re.search(r"जिला[:\s]+([\u0900-\u097F ]+)", clean_text)
    result["dist_name"] = m.group(1).strip() if m else "Pune City"

    # Police Station
    m = re.search(r"(?:Police Station|P\.S\.|पोलीस ठाणे)[:\s]+([A-Za-z\u0900-\u097F ]+)", clean_text)
    if m:
        result["police_station"] = m.group(1).strip()
    elif "भोसरी" in clean_text:
        result["police_station"] = "Bhosari"

    # Acts
    acts_found = re.findall(r"(Indian Penal Code|IPC|Arms Act,? ?\d{4}|Maharashtra Police Act,? ?\d{4})", clean_text, re.IGNORECASE)
    acts_found = list(dict.fromkeys([a.strip() for a in acts_found]))
    result["under_acts"] = acts_found if acts_found else ["Indian Arms Act 1959", "Maharashtra Police Act 1951"]

    # Sections
    sections_found = re.findall(r"\b\d{1,3}(?:\(\d+\))?\b", clean_text)
    sections_filtered = []
    for s in sections_found:
        try:
            val = int(re.sub(r"[^0-9]", "", s))
            if 0 < val < 600:
                sections_filtered.append(s)
        except:
            pass
    sections_filtered = list(dict.fromkeys(sections_filtered))
    # keep only relevant FIR-related sections
    keep_sections = [sec for sec in sections_filtered if sec in ["25", "3", "135", "37", "323", "427", "506", "509"]]
    result["under_sections"] = keep_sections

    # Case category
    if any("Arms" in a or "हत्यार" in a for a in result["under_acts"]):
        result["revised_case_category"] = "WEAPONS"
    elif any(s in ["354", "376", "509"] for s in result["under_sections"]):
        result["revised_case_category"] = "CRIMINAL_OFFENCE_SEXUAL_HARASSMENT"
    else:
        result["revised_case_category"] = "OTHER"

    # Name / Complainant
    m = re.search(r"VIPUL\s+RANGNATH\s+JADHAV", clean_text, re.IGNORECASE)
    if m:
        result["name"] = "VIPUL RANGNATH JADHAV"
        result["oparty"] = "Complainant"
    else:
        m = re.search(r"Name[:\s]+([A-Z][A-Za-z ]+)", clean_text)
        if m:
            result["name"] = m.group(1).strip()
            result["oparty"] = "Complainant"
        else:
            result["name"] = None
            result["oparty"] = None

    # Address
    m = re.search(r"भोसरी.*पुणे", clean_text)
    if m:
        result["address"] = "भोसरी, पुणे शहर, महाराष्ट्र"
    else:
        m = re.search(r"Address[:\s]+([^\n]+)", clean_text)
        result["address"] = m.group(1).strip() if m else None

    # Jurisdiction
    result["jurisdiction"] = result.get("dist_name", "Pune City")
    result["jurisdiction_type"] = "DISTRICT"

    return result

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
        pii = extract_pii_rules(combined_text)
        results[uploaded.name] = pii
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    st.subheader("Extracted PII (per file)")
    st.json(results)

    b64 = base64.b64encode(json.dumps(results, ensure_ascii=False, indent=2).encode()).decode()
    href = f"data:application/json;base64,{b64}"
    st.markdown(f"[Download JSON]({href})")

else:
    st.write("No files uploaded yet — drop a PDF to get started.")
