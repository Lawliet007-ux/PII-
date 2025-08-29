# app.py
import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import io
import re
import json
import os
from datetime import datetime
from collections import defaultdict

# NLP libs
import stanza
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# OCR libs
try:
    import pytesseract
    from PIL import Image
    import cv2
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Legal FIR PII Extractor", layout="wide")

# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
DATE_PATTERNS = [
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b',
]

PHONE_REGEX = re.compile(r'(?:\+91[\-\s]?)?(\d{10})\b')
AADHAAR_REGEX = re.compile(r'\b\d{4}\s*\d{4}\s*\d{4}\b|\b\d{12}\b')
PAN_REGEX = re.compile(r'\b([A-Z]{5}\d{4}[A-Z])\b', re.IGNORECASE)
FIR_REGEX = re.compile(r'\bFIR\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/\-]+)\b', re.IGNORECASE)
MONEY_REGEX = re.compile(r'\b[₹Rs\.]?\s?[\d,]+(?:\.\d{1,2})?\b')

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
            # basic thresholding for better OCR
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
        entities.append({"text": ent.text, "type": ent.type, "start_char": ent.start_char, "end_char": ent.end_char})
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
    output = {"names": set(), "phones": set(), "dates": set(), "ids": set(), "addresses": set(), "roles": set(), "fir_no": set(), "money": set()}
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
    # process NER
    for ent in ner_entities:
        t = ent['type']
        txt = ent['text'].strip()
        if t in ('PERSON','PER','PERSON_NAME'):
            output['names'].add(txt)
        elif t in ('ORG','GPE','LOC','LOCATION','CITY'):
            output['addresses'].add(txt)
        elif t in ('DATE','TIME'):
            output['dates'].add(txt)
        else:
            # keep other entities
            pass
    # Heuristics to find roles like IO, Complainant, Inspector
    # Look for patterns
    role_patterns = [
        (r'Investigating Officer[:\-\s]*([A-Z\s]+)', 'investigating_officer'),
        (r'Name\s*\(नाव\):\s*([A-Z\s]+)', 'complainant'),
        (r'Inspector[:\-\s]*([A-Z\s]+)', 'inspector'),
    ]
    for pat, role in role_patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            output['roles'].add((role, m.group(1).strip()))
    # convert sets to lists
    return {k: list(v) for k,v in output.items()}

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("PII Extractor — Legal / FIR (Hindi + English)")

uploaded = st.file_uploader("Upload a PDF (FIR / legal doc)", type=['pdf'])
if uploaded:
    file_bytes = uploaded.read()
    st.sidebar.header("Options")
    use_ocr = st.sidebar.checkbox("Force OCR (if PDF is scanned)", value=False)
    st.sidebar.info("The app will try text extraction first and fall back to OCR if needed.")

    # Try text extraction
    text1 = extract_text_pymupdf(file_bytes)
    if not text1.strip():
        text1 = extract_text_pdfplumber(file_bytes)
    # fallback to OCR if requested or if text is empty
    if use_ocr or (not text1.strip()):
        # lazy import
        import numpy as np
        if OCR_AVAILABLE:
            st.info("Running OCR (requires Tesseract installed)...")
            ocr_text = ocr_pdf(file_bytes)
            text = (text1 + "\n" + ocr_text).strip()
        else:
            st.error("OCR libraries not available in runtime.")
            text = text1
    else:
        text = text1

    st.subheader("Preview of extracted text (first 4000 chars)")
    st.text_area("Document text", value=text[:4000], height=300)

    # language detection
    try:
        lang = detect(text[:5000]) if text.strip() else 'en'
    except:
        lang = 'en'
    st.write("Detected language (approx):", lang)

    # Run stanza NER (will download models first time)
    with st.spinner("Running NER and regex extractors..."):
        # ensure stanza models are present
        if not os.path.exists(os.path.expanduser('~/stanza_resources')):
            stanza.download('en', processors='tokenize,ner')
            stanza.download('hi', processors='tokenize,ner')
        # run both hi and en and merge entities for maximum recall
        ner_en = run_stanza_ner(text, 'en')
        ner_hi = run_stanza_ner(text, 'hi')
        ner_entities = ner_en + ner_hi

        regex_res = find_regex_all(text)
        merged = merge_results(regex_res, ner_entities, text)

    st.subheader("Extracted PII (structured)")
    st.json(merged)

    # show downloadable JSON
    st.download_button("Download result (JSON)", data=json.dumps(merged, ensure_ascii=False, indent=2), file_name="pii_extraction.json", mime="application/json")

    st.info("If something looks wrong, use the manual review section below to correct entries (recommended for legal use).")
    with st.expander("Manual review / corrections"):
        corrected_json = st.text_area("Edit JSON", value=json.dumps(merged, ensure_ascii=False, indent=2), height=300)
        if st.button("Save corrected JSON"):
            try:
                parsed = json.loads(corrected_json)
                st.success("Saved (in memory). Download below.")
                st.download_button("Download corrected JSON", data=json.dumps(parsed, ensure_ascii=False, indent=2), file_name="pii_extraction_corrected.json", mime="application/json")
            except Exception as e:
                st.error("Invalid JSON: " + str(e))
