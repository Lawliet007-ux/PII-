import streamlit as st
import re
import json
from transformers import pipeline

st.set_page_config(page_title="Advanced PII Extractor", layout="wide")

# Regex patterns (same as before, extended)
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

# IndicNER model (better than XLM-R for Indian PII)
@st.cache_resource
def load_ner():
    return pipeline("ner", model="ai4bharat/IndicNER", aggregation_strategy="simple")

ner_model = load_ner()

# --- Regex extractor ---
def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group().strip(), "confidence": 0.99})
    return out

# --- NER extractor ---
def ner_extract(text):
    ents = ner_model(text)
    out = []
    for ent in ents:
        if ent["entity_group"] in ["PER", "LOC", "ORG"]:
            out.append({
                "label": ent["entity_group"].lower(),
                "text": ent["word"],
                "confidence": float(ent["score"])
            })
    return out

# --- LLM-based structured extraction ---
from transformers import pipeline as hf_pipeline
llm = hf_pipeline("text2text-generation", model="google/flan-t5-large")  # lightweight

def llm_extract(text):
    prompt = f"""
    Extract all Personal Identifiable Information (PII) from the following text.
    Return JSON with keys: Names, Addresses, IDs, Contacts, PoliceStations, Dates.
    
    Text: {text}
    """
    response = llm(prompt, max_new_tokens=512)[0]["generated_text"]
    try:
        data = json.loads(response)
        out = []
        for key, vals in data.items():
            for v in vals:
                out.append({"label": key.lower(), "text": v, "confidence": 0.95})
        return out
    except:
        return []

# --- Merge ---
def extract_pii(text):
    regex_hits = regex_extract(text)
    ner_hits = ner_extract(text)
    llm_hits = llm_extract(text)
    return regex_hits + ner_hits + llm_hits

# --- Streamlit UI ---
st.title("🔎 Advanced PII Extractor (Regex + IndicNER + LLM)")
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
