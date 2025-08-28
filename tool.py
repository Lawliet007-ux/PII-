import streamlit as st
import re
import json

st.set_page_config(page_title="PII Extractor", layout="wide")

# ---------------- Regex Patterns ----------------
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

# ---------------- Streamlit UI ----------------
st.title("🔎 PII Extractor (Safe Mode)")
text = st.text_area("Paste FIR / judgment text:", height=200)

if st.button("Extract PII"):
    if not text.strip():
        st.warning("Please enter some text first!")
    else:
        results = regex_extract(text)

        if results:
            st.success(f"Found {len(results)} PII items")
            st.json(results)
        else:
            st.error("No PII found with regex rules.")

st.markdown("---")
st.info("⚡ Currently running in **safe regex-only mode** (no heavy ML models).")
