# app_stanza.py
import streamlit as st
import re, json
import pandas as pd
from dateutil import parser as dateparser

# --- NLP (Stanza) ---
import stanza

@st.cache_resource
def load_stanza_pipelines():
    # Download once: stanza.download("en"); stanza.download("hi")
    return {
        "en": stanza.Pipeline("en", processors="tokenize,ner", use_gpu=False),
        "hi": stanza.Pipeline("hi", processors="tokenize,ner", use_gpu=False)
    }

nlp_pipes = load_stanza_pipelines()

# --- Regex patterns ---
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
}

def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group(), "confidence": 0.95})
    return out

def stanza_extract(text, lang="en"):
    out = []
    if lang not in nlp_pipes:
        return out
    doc = nlp_pipes[lang](text)
    for ent in doc.ents:
        out.append({"label": ent.type, "text": ent.text, "confidence": 0.9})
    return out

# --- Main extractor ---
def extract_pii(text: str):
    results = []
    # regex (lang agnostic)
    results.extend(regex_extract(text))
    # run both Hindi + English NER (since mix is possible)
    results.extend(stanza_extract(text, "hi"))
    results.extend(stanza_extract(text, "en"))
    # deduplicate
    seen = set()
    final = []
    for r in results:
        key = (r["label"].lower(), r["text"].lower())
        if key not in seen:
            seen.add(key)
            final.append(r)
    return final

# --- Streamlit UI ---
st.title("PII Extractor (Regex + Stanza NER)")
txt = st.text_area("Paste OCR/ES legal text here:", height=250)

if st.button("Extract PII"):
    if not txt.strip():
        st.error("Please enter text")
    else:
        ents = extract_pii(txt)
        if ents:
            df = pd.DataFrame(ents)
            st.dataframe(df)
            st.download_button("Download JSON", data=json.dumps(ents, ensure_ascii=False, indent=2),
                               file_name="pii.json", mime="application/json")
        else:
            st.warning("No PII detected.")
