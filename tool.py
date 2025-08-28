# tool.py
import streamlit as st
import re
import spacy
from transformers import pipeline

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    pii_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return nlp, pii_model

nlp, pii_model = load_models()

# ---------- Regex Patterns ----------
regex_patterns = {
    "phone": r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",
    "date": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "passport": r"\b[A-Z]{1}-?\d{7}\b",
}

# ---------- Extract PII ----------
def extract_pii(text):
    results = []

    # Regex-based PII
    for label, pattern in regex_patterns.items():
        for match in re.finditer(pattern, text):
            results.append({
                "label": label,
                "text": match.group(),
                "confidence": 1.0
            })

    # spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            results.append({
                "label": ent.label_.lower(),
                "text": ent.text,
                "confidence": 0.95
            })

    # Transformer NER (chunking for long texts)
    max_chunk = 400
    words = text.split()
    for i in range(0, len(words), max_chunk):
        chunk = " ".join(words[i:i+max_chunk])
        ner_results = pii_model(chunk)
        for r in ner_results:
            results.append({
                "label": r["entity_group"].lower(),
                "text": r["word"],
                "confidence": float(r["score"])
            })

    # Deduplicate
    unique = []
    seen = set()
    for r in results:
        key = (r["label"], r["text"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique

# ---------- Streamlit UI ----------
st.set_page_config(page_title="PII Extractor", layout="wide")
st.title("📑 PII Extraction from FIR / Legal Text")

input_text = st.text_area("Paste the FIR/legal case text below:", height=300)

if st.button("Extract PII"):
    if input_text.strip():
        with st.spinner("Extracting..."):
            pii_data = extract_pii(input_text)
        st.subheader("📑 Extracted PII")
        for item in pii_data:
            st.json(item)
    else:
        st.warning("Please enter some text.")
