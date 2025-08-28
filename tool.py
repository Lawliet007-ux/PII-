import streamlit as st
import re
import json
from transformers import pipeline

st.set_page_config(page_title="Advanced PII Extractor", layout="wide")

# ------------------- Regex Patterns -------------------
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"(?:FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+)",
    "cnr": r"\bCNR\s*No\.?\s*[:\s]*[A-Z0-9-]+\b",
    "vehicle_no": r"\b[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}\b",
    "passport": r"\b[A-Z]\d{7}\b",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "pin": r"\b\d{6}\b"
}

def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group().strip(), "confidence": 0.99})
    return out

# ------------------- NER Model -------------------
@st.cache_resource
def load_ner():
    # Using a smaller multilingual model (lightweight)
    return pipeline("ner", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")

# ------------------- Extraction -------------------
def extract_pii(text):
    results = []

    # Regex Pass
    results.extend(regex_extract(text))

    # NER Pass
    try:
        ner = load_ner()
        ents = ner(text)
        for ent in ents:
            if ent["entity_group"] in ["PER", "LOC", "ORG"]:
                results.append({
                    "label": ent["entity_group"].lower(),
                    "text": ent["word"],
                    "confidence": float(ent["score"])
                })
    except Exception as e:
        st.error(f"NER model failed: {e}")

    return results

# ------------------- Streamlit UI -------------------
st.title("🔎 Advanced PII Extractor")
txt = st.text_area("Paste FIR / Legal text", height=200)

if st.button("Extract PII"):
    if not txt.strip():
        st.warning("Paste some text first!")
    else:
        results = extract_pii(txt)
        if results:
            for r in results:
                st.json(r)
        else:
            st.error("No PII found.")
