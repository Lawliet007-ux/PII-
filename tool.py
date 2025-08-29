import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import io
import re
import json
import os
from datetime import datetime
from collections import defaultdict

# NLP
import stanza
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# OCR
try:
    import pytesseract
    from PIL import Image
    import numpy as np
    import cv2
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ----------------------------
# Regex patterns
# ----------------------------
DATE_PATTERNS = [
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',
]
PHONE_REGEX = re.compile(r'(?:\+91[\-\s]?)?(\d{10})\b')
AADHAAR_REGEX = re.compile(r'\b\d{4}\s*\d{4}\s*\d{4}\b|\b\d{12}\b')
PAN_REGEX = re.compile(r'\b([A-Z]{5}\d{4}[A-Z])\b', re.IGNORECASE)
FIR_REGEX = re.compile(r'\bFIR\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/\-]+)\b', re.IGNORECASE)
MONEY_REGEX = re.compile(r'\b[₹Rs\.]?\s?[\d,]+(?:\.\d{1,2})?\b')


# ----------------------------
# Helpers
# ----------------------------
def extract_text_pymupdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text)


def extract_text_pdfplumber(file_bytes):
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            text.append(p.extract_text() or "")
    return "\n".join(text)


def ocr_pdf(file_bytes):
    if not OCR_AVAILABLE:
        return ""
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            pil = p.to_image(resolution=300).original
            gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_t = pytesseract.image_to_string(th, lang='hin+eng')
            text.append(ocr_t)
    return "\n".join(text)


def run_stanza_ner(text, lang):
    if lang.startswith('hi'):
        nlp = stanza.Pipeline(lang='hi', processors='tokenize,ner', use_gpu=False, logging_level='ERROR')
    else:
        nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', use_gpu=False, logging_level='ERROR')
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "type": ent.type})
    return entities


def find_regex_all(text):
    res = defaultdict(list)
    for pat in DATE_PATTERNS:
        for m in re.findall(pat, text):
            res['dates'].append(m)
    for m in PHONE_REGEX.findall(text):
        res['phones'].append(m)
    for m in AADHAAR_REGEX.findall(text):
        res['aadhaar'].append(m)
    for m in PAN_REGEX.findall(text):
        res['pan'].append(m)
    for m in FIR_REGEX.findall(text):
        res['fir_no'].append(m)
    for m in MONEY_REGEX.findall(text):
        res['money'].append(m)
    return res


def normalize_date(s):
    for fmt in ("%d/%m/%Y","%d/%m/%y","%d-%m-%Y","%d-%m-%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except:
            pass
    return s


def merge_results(regex_res, ner_entities, raw_text):
    output = {"names": set(), "phones": set(), "dates": set(), "ids": set(),
              "addresses": set(), "roles": set(), "fir_no": set(), "money": set()}

    for ph in regex_res.get('phones', []):
        output['phones'].add(ph)
    for a in regex_res.get('aadhaar', []):
        output['ids'].add(("aadhaar", a))
    for p in regex_res.get('pan', []):
        output['ids'].add(("pan", p))
    for f in regex_res.get('fir_no', []):
        output['fir_no'].add(f)
    for m in regex_res.get('money', []):
        output['money'].add(m)
    for d in regex_res.get('dates', []):
        output['dates'].add(normalize_date(d))

    for ent in ner_entities:
        t, txt = ent['type'], ent['text'].strip()
        if t in ('PERSON','PER'):
            output['names'].add(txt)
        elif t in ('ORG','GPE','LOC','LOCATION'):
            output['addresses'].add(txt)
        elif t in ('DATE','TIME'):
            output['dates'].add(txt)

    return {k: list(v) for k,v in output.items()}


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Robust FIR / Legal PII Extractor (Hindi + English / Mixed)")

uploaded = st.file_uploader("Upload a PDF (FIR / legal document)", type=['pdf'])
if uploaded:
    file_bytes = uploaded.read()
    use_ocr = st.sidebar.checkbox("Force OCR (for scanned PDFs)", value=False)

    # Try extraction
    text = extract_text_pymupdf(file_bytes)
    if not text.strip():
        text = extract_text_pdfplumber(file_bytes)
    if use_ocr or (not text.strip()):
        if OCR_AVAILABLE:
            st.info("Running OCR...")
            text = ocr_pdf(file_bytes)
        else:
            st.error("OCR not available. Install Tesseract with Hindi+English models.")
    
    st.subheader("Extracted Text Preview")
    st.text_area("Text", text[:4000], height=300)

    try:
        lang = detect(text[:1000]) if text.strip() else 'en'
    except:
        lang = 'en'
    st.write("Detected language:", lang)

    with st.spinner("Extracting PII..."):
        if not os.path.exists(os.path.expanduser('~/stanza_resources')):
            stanza.download('en')
            stanza.download('hi')
        ner_entities = run_stanza_ner(text, 'en') + run_stanza_ner(text, 'hi')
        regex_res = find_regex_all(text)
        merged = merge_results(regex_res, ner_entities, text)

    st.subheader("🔎 Extracted PII")
    st.json(merged)

    st.download_button("⬇️ Download JSON", 
                       data=json.dumps(merged, ensure_ascii=False, indent=2), 
                       file_name="pii_extraction.json", 
                       mime="application/json")
