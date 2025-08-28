# app_transformer.py
import streamlit as st
import re, json
import pandas as pd
from transformers import pipeline
from typing import List, Dict

# ---------------------------
# Load HuggingFace NER model
# ---------------------------
@st.cache_resource
def load_model():
    # ai4bharat/IndicNER supports Hindi + English
    return pipeline("ner", model="ai4bharat/IndicNER", aggregation_strategy="simple")

ner_model = load_model()

# ---------------------------
# Regex patterns for structured PII
# ---------------------------
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "pin": r"\b\d{6}\b"
}

def regex_extract(text: str) -> List[Dict]:
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group(), "confidence": 0.95})
    return out

# ---------------------------
# NER Extract (Transformer)
# ---------------------------
def ner_extract(text: str) -> List[Dict]:
    results = []
    # Split into chunks (model max length ~512 tokens)
    words = text.split()
    chunk_size = 200
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        ents = ner_model(chunk)
        for e in ents:
            results.append({
                "label": e["entity_group"],
                "text": e["word"],
                "confidence": float(e["score"])
            })
    return results

# ---------------------------
# Merge + dedup
# ---------------------------
def merge_results(results: List[Dict]) -> List[Dict]:
    seen = set()
    final = []
    for r in results:
        key = (r["label"].lower(), r["text"].lower())
        if key not in seen:
            seen.add(key)
            final.append(r)
    return final

# ---------------------------
# Full extractor
# ---------------------------
def extract_pii(text: str) -> List[Dict]:
    regex_hits = regex_extract(text)
    ner_hits = ner_extract(text)
    merged = merge_results(regex_hits + ner_hits)
    return merged

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("⚖️ PII Extractor (Regex + Transformer NER)")

txt = st.text_area("Paste OCR/Elasticsearch FIR/legal text here:", height=300)

if st.button("Extract PII"):
    if not txt.strip():
        st.error("Please enter text")
    else:
        ents = extract_pii(txt)
        if ents:
            df = pd.DataFrame(ents)
            st.dataframe(df)
            st.download_button("Download JSON",
                               data=json.dumps(ents, ensure_ascii=False, indent=2),
                               file_name="pii.json",
                               mime="application/json")
        else:
            st.warning("No PII detected.")
