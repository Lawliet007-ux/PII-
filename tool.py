"""
Streamlit Multilingual PII Extractor for Legal OCR Text (EN + Indic languages)
-----------------------------------------------------------------------------
"""

import json
import re
from typing import Dict, List, Tuple, Any

import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ----------------------------
# Normalization
# ----------------------------

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def normalize_ocr(text: str) -> str:
    if not text:
        return ""
    t = text
    t = t.translate(DEVANAGARI_DIGITS)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ----------------------------
# Regex Rules
# ----------------------------

RE_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
RE_PHONE = re.compile(r"(?<!\d)(?:\+?91[- ]?)?(?:[6-9]\d{9})(?!\d)")
RE_DATE = re.compile(r"\b(?:(?:[0-3]?\d)[/-](?:0?\d|1[0-2])[/-](?:\d{2,4})|\d{4}-\d{2}-\d{2})\b")

RULES = [
    ("EMAIL", RE_EMAIL),
    ("PHONE", RE_PHONE),
    ("DATE", RE_DATE)
]

# ----------------------------
# Load NER
# ----------------------------

@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_name: str = "ai4bharat/IndicNER"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "ADDRESS": "ADDRESS",
}

# ----------------------------
# Extraction
# ----------------------------

def run_regex_rules(text: str):
    spans = []
    for label, pat in RULES:
        for m in pat.finditer(text):
            spans.append({"label": label, "text": m.group(0)})
    return spans

def run_ner(text: str, nlp, min_score: float):
    spans = []
    outputs = nlp(text)
    for ent in outputs:
        label = LABEL_MAP.get(ent.get("entity_group"), None)
        if not label:
            continue
        score = float(ent.get("score", 0.0))
        if score < min_score:
            continue
        spans.append({"label": label, "text": text[ent["start"]:ent["end"]], "score": round(score, 3)})
    return spans

# ----------------------------
# Streamlit App
# ----------------------------

st.set_page_config(page_title="Multilingual PII Extractor (Legal OCR)", layout="wide")
st.title("🔎 Multilingual PII Extractor for Legal OCR (EN + Indic)")

with st.sidebar:
    st.header("Model & Settings")
    model_name = st.text_input("HF model", value="ai4bharat/IndicNER")
    threshold = st.slider("NER confidence threshold", 0.0, 1.0, 0.55, 0.01)
    apply_regex = st.checkbox("Apply regex rules", value=True)
    apply_ner = st.checkbox("Apply NER model", value=True)
    mask_token = st.text_input("Mask token for redaction", value="▮▮▮")

nlp = load_ner_pipeline(model_name)

sample_text = (
    "P.S. (Police Thane): Bhosari \nFIR No.: 0523 \nDate and Time of FIR: 19/11/2017 at 21:33 \nDistrict: Pune City"
)

text = st.text_area("Paste text here", value=sample_text, height=200)

col_run, col_clear = st.columns([1, 1])
with col_run:
    run = st.button("Extract PII", type="primary")
with col_clear:
    clear = st.button("Clear")

if run:
    norm = normalize_ocr(text)
    spans = []
    if apply_regex:
        spans.extend(run_regex_rules(norm))
    if apply_ner:
        spans.extend(run_ner(norm, nlp, threshold))

    st.subheader("Results")
    st.json(spans)

elif clear:
    st.experimental_rerun()

st.markdown("---")
st.caption("This tool extracts PII from multilingual OCR text using regex + NER.")
