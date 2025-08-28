# pii_extractor.py
import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ------------------- Setup -------------------
st.set_page_config(page_title="PII Extractor", layout="wide")

MODEL_NAME = "dslim/bert-base-NER"   # Swap with fine-tuned model if available

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

nlp = load_model()

# ------------------- Regex Rules -------------------
regex_patterns = {
    "phone": r"\b[6-9]\d{9}\b",
    "fir_no": r"\bFIR\s*No[:\-]?\s*\w+",
    "date": r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
    "time": r"\b\d{1,2}:\d{2}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "pincode": r"\b\d{6}\b"
}

def regex_extract(text):
    """Extract PII using regex rules"""
    results = []
    for label, pattern in regex_patterns.items():
        matches = re.findall(pattern, text)
        for m in matches:
            results.append({
                "label": label,
                "text": m,
                "confidence": 1.0  # regex = high confidence
            })
    return results

# ------------------- NER Extract -------------------
def ner_extract(text):
    """Extract PII using NER model with filtering"""
    ents = nlp(text)
    results = []
    for e in ents:
        word = e["word"].strip()
        # Filter out junk: single chars, weird subtokens
        if len(word) < 2: 
            continue
        results.append({
            "label": e["entity_group"],
            "text": word,
            "confidence": round(float(e["score"]), 3)
        })
    return results

# ------------------- Merge + Deduplicate -------------------
def merge_results(regex_res, ner_res):
    seen = set()
    final = []
    for r in regex_res + ner_res:
        key = (r["label"], r["text"].lower())
        if key not in seen:
            seen.add(key)
            final.append(r)
    return final

# ------------------- UI -------------------
st.title("🔍 Robust PII Extractor for FIR / Legal Docs")

sample_text = """FIR No: 0523, Date: 19/11/2017, Time: 21:33,
Police Station: Bhosari, District: Pune,
Phone: 7720010466, Name: Anand Jes, Location: Pune City, Maharashtra, India"""

text_input = st.text_area("Paste FIR / Document text here:", sample_text, height=300)

if st.button("Extract PII"):
    with st.spinner("Extracting PII..."):
        regex_res = regex_extract(text_input)
        ner_res = ner_extract(text_input)
        results = merge_results(regex_res, ner_res)

    if results:
        st.subheader("📑 Extracted PII")
        for r in results:
            st.json(r)
    else:
        st.warning("No PII found in the text!")
