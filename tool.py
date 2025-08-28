# tool.py
import streamlit as st
import re
from rapidfuzz import fuzz

st.set_page_config(page_title="PII Extractor", layout="wide")

# ------------------- Regex Patterns -------------------
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"(?:FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+)",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "pin": r"\b\d{6}\b"
}

# ------------------- Gazetteer -------------------
GAZETTEER = {
    "state": ["Maharashtra", "Delhi", "Karnataka", "Goa"],
    "district": ["Pune", "Nagpur", "Mumbai", "Bhosari"],
    "police": ["Police Station", "PS", "Thane", "Bhosari Police Station"]
}

def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group().strip(), "confidence": 0.99})
    return out

def gazetteer_extract(text):
    out = []
    for label, words in GAZETTEER.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
                out.append({"label": label, "text": w, "confidence": 0.98})
    return out

def merge_results(*sources):
    final = []
    for src in sources:
        for r in src:
            if not any(fuzz.ratio(r["text"].lower(), f["text"].lower()) > 90 for f in final):
                final.append(r)
    return final

def extract_pii(text):
    regex_hits = regex_extract(text)
    gazette_hits = gazetteer_extract(text)
    return merge_results(regex_hits, gazette_hits)

# ------------------- Streamlit UI -------------------
st.title("🔎 PII Extractor (Hybrid Model)")
st.write("Extract Aadhaar, PAN, Phone, Email, FIR No., Date/Time, PIN, Gazetteer locations etc.")

user_input = st.text_area("Paste FIR / Legal Text Here", height=200)

if st.button("Extract PII"):
    if not user_input.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        results = extract_pii(user_input)
        if results:
            st.success(f"✅ Found {len(results)} PII entities")
            for r in results:
                st.json(r)
        else:
            st.error("❌ No PII found in the text.")
