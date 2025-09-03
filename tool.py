# app_streamlit_pii_extractor.py
import streamlit as st
import pdfplumber
import fitz
import re
import json
import io
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, Any, List

# Optional OCR
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor (Hindi + English)", layout="wide")

st.title("FIR PII Extractor — Hindi + English (Streamlit)")
st.write("Upload one or more FIR PDFs (Hindi / Marathi / English). The tool will extract common PII fields.")

uploaded = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

# --- extraction helpers (same approach as used on your samples) ---
def extract_text_pdfplumber_bytes(file_bytes: bytes) -> Tuple[str,str]:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages).strip()
        return text, "pdfplumber"
    except Exception:
        return "", "pdfplumber_failed"

def extract_text_fitz_bytes(file_bytes: bytes) -> Tuple[str,str]:
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for p in doc:
            pages.append(p.get_text("text") or "")
        doc.close()
        return "\n".join(pages).strip(), "fitz_text"
    except Exception:
        return "", "fitz_failed"

def ocr_with_fitz_and_tesseract_bytes(file_bytes: bytes, lang="hin+eng") -> Tuple[str,str]:
    if not OCR_AVAILABLE:
        return "", "ocr_not_available"
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                t = pytesseract.image_to_string(img, lang=lang)
            except Exception:
                t = pytesseract.image_to_string(img)
            texts.append(t)
        doc.close()
        return "\n".join(texts).strip(), "ocr_tesseract"
    except Exception:
        return "", "ocr_failed"

def extract_text_bytes(file_bytes: bytes, prefer_ocr=False) -> Tuple[str,str]:
    if prefer_ocr:
        o, m = ocr_with_fitz_and_tesseract_bytes(file_bytes)
        return o, m
    # try pdfplumber
    t, m = extract_text_pdfplumber_bytes(file_bytes)
    if len(t) > 200:
        return t, m
    t2, m2 = extract_text_fitz_bytes(file_bytes)
    if len(t2) > 200:
        return t2, m2
    # fallback to OCR if available
    t3, m3 = ocr_with_fitz_and_tesseract_bytes(file_bytes)
    if len(t3) > 50:
        return t3, m3
    # any combined
    combined = (t or "") + "\n" + (t2 or "")
    return combined.strip(), "combined"

# Parsing heuristics (kept compact — extendable)
def extract_year(text: str) -> str:
    m = re.search(r'Year\s*[:\-]?\s*(\d{4})', text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'वष[:\)\s\-]*\s*(\d{4})', text)
    if m:
        return m.group(1)
    m = re.search(r'\b(20\d{2}|19\d{2})\b', text)
    return m.group(1) if m else ""

def extract_fir_no(text: str) -> str:
    m = re.search(r'FIR\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\-\/]+)', text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'थम खबर .*[:\-]?\s*([A-Za-z0-9\-\/]+)', text)
    if m:
        return m.group(1)
    return ""

def extract_field_labelled(text: str, english_label: str, hindi_label: str="") -> str:
    # generic finder for patterns like 'Name (नाव): <value>' or 'Name: <value>'
    pat1 = rf'{re.escape(english_label)}\s*\(.*?\)\s*[:\-]?\s*([^\n\r]+)'
    m = re.search(pat1, text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    pat2 = rf'{re.escape(english_label)}\s*[:\-]?\s*([^\n\r]+)'
    m = re.search(pat2, text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if hindi_label:
        pat3 = rf'{re.escape(hindi_label)}\s*[:\-]?\s*([^\n\r]+)'
        m = re.search(pat3, text)
        if m:
            return m.group(1).strip()
    return ""

def extract_acts_sections(text: str):
    acts = []
    sections = []
    # look for 'Acts' block
    m = re.search(r'Acts\s*\(.*?\)\s*[:\-]?\s*(.*?)\n', text, re.IGNORECASE)
    if m:
        acts.append(m.group(1).strip())
    # look for 'अधिननयम' near acts
    m = re.search(r'अधिननयम\).*?\n(.*?)\n', text)
    if m:
        acts.append(m.group(1).strip())
    # sections look
    sections_found = re.findall(r'\b(\d{2,3}(?:\(\d+\))?(?:,\s*\d{1,3}(?:\(\d+\))?)*)\b', text)
    # filter out years
    sections = [s for s in sections_found if not re.match(r'20\d{2}', s)]
    return list(dict.fromkeys(acts)), list(dict.fromkeys(sections))

def extract_names_phones(text: str):
    names = []
    phones = []
    m = re.findall(r'Name\s*\(नाव\)\s*[:\-]?\s*([^\n\r]+)', text, re.IGNORECASE)
    names.extend(m)
    m2 = re.findall(r'नाव\)\s*[:\-]?\s*([^\n\r]+)', text)
    names.extend(m2)
    # phone patterns
    phones.extend(re.findall(r'91[- ]?\d{10}\b', text))
    phones.extend(re.findall(r'\b\d{10}\b', text))
    return list(dict.fromkeys([n.strip() for n in names])), list(dict.fromkeys(phones))

def extract_address_candidates(text: str):
    addrs = []
    m = re.findall(r'Address\s*\(.*?\)\s*[:\-]?\s*([^\n\r]+)', text, re.IGNORECASE)
    addrs.extend(m)
    m = re.findall(r'प ा\)\s*[:\-]?\s*([^\n\r]+)', text)
    addrs.extend(m)
    # fallback longer lines containing keywords
    for line in text.splitlines():
        if len(line) > 30 and any(k in line for k in ['नगर','गन','लॉट','पत्ता','पोलस','पोलीस','नगर', 'PUNE','MAH']):
            addrs.append(line.strip())
    return list(dict.fromkeys(addrs))[:3]

def infer_jurisdiction_type(state_text: str):
    if not state_text:
        return "UNKNOWN"
    return "STATE_LEVEL"

def normalize_case_category(acts, sections, text):
    for s in sections:
        if re.search(r'\b\d{2,3}\b', s):
            return "CRIMINAL"
    for a in acts:
        if 'B.N.S.S' in a or 'बी एन एस' in a:
            return "SPECIAL"
    if '154' in text or 'Cr.P.C' in text or 'Cr.P.C.' in text:
        return "CRIMINAL"
    return "OTHER"

def parse_fields(text: str) -> Dict[str,Any]:
    r = {}
    r['year'] = extract_year(text)
    r['fir_no'] = extract_fir_no(text)
    r['police_station'] = extract_field_labelled(text, "P.S.", "पोलीस ठाणे")
    r['dist_name'] = extract_field_labelled(text, "District", "िज हा")
    r['state_name'] = extract_field_labelled(text, "State", "रा य") or ("Maharashtra" if "महारा" in text or "महा" in text else "")
    acts, sections = extract_acts_sections(text)
    r['under_acts'] = acts
    r['under_sections'] = sections
    names, phones = extract_names_phones(text)
    r['name'] = names
    r['phone'] = phones
    r['address'] = extract_address_candidates(text)
    r['jurisdiction_type'] = infer_jurisdiction_type(r['state_name'])
    r['revised_case_category'] = normalize_case_category(acts, sections, text)
    return r

# --- UI options and processing ---
prefer_ocr = st.checkbox("Force OCR (use if text-extraction fails or PDF is scanned)", value=False)
lang_option = st.selectbox("OCR language (if using OCR)", ["hin+eng", "eng", "hin"]) if OCR_AVAILABLE else st.info("OCR not available in this runtime")

if uploaded:
    st.markdown("### Extraction results")
    results = {}
    for up in uploaded:
        raw = up.read()
        text, method = extract_text_bytes(raw, prefer_ocr)
        parsed = parse_fields(text)
        results[up.name] = {"meta": {"extraction_method": method}, "fields": parsed}
        st.subheader(up.name)
        st.write("**Extraction method**:", method)
        st.json(parsed)
    # allow download of JSON
    st.download_button("Download all results (JSON)", data=json.dumps(results, ensure_ascii=False, indent=2), file_name="pii_extraction_results.json", mime="application/json")
else:
    st.info("Upload PDFs to start extraction. You can upload multiple FIR PDFs at once.")

st.markdown("---")
st.write("Notes / Next steps: you can improve accuracy by training a custom NER (spaCy) for Hindi/Marathi + English labels, adding more regex rules tuned to state language variants, and adding a verification UI to mark/correct extracted fields.")
