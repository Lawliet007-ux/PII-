# robust_pii_extractor.py
import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ------------------- Setup -------------------
st.set_page_config(page_title="Robust PII Extractor", layout="wide")

MODEL_NAME = "dslim/bert-base-NER"   # can replace with "ai4bharat/IndicNER" for Indian names

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
    "fir_no": r"\bFIR\s*No[:\-]?\s*[\w\/\-]+",
    "date": r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
    "time": r"\b\d{1,2}:\d{2}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "pincode": r"\b\d{6}\b",
    "vehicle_no": r"\b[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}\b",
    "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b"
}

def regex_extract(text):
    results = []
    for label, pattern in regex_patterns.items():
        for m in re.findall(pattern, text):
            results.append({"label": label, "text": m.strip(), "confidence": 1.0})
    return results

# ------------------- NER Extract -------------------
def ner_extract(text):
    ents = nlp(text)
    results = []
    for e in ents:
        word = e["word"].strip()

        # filter out garbage
        if len(word) < 2:
            continue
        if e["score"] < 0.6:   # drop low confidence
            continue
        if word.startswith("##"):   # join sub-tokens
            continue

        results.append({
            "label": e["entity_group"].lower(),  # LOC → loc
            "text": word,
            "confidence": round(float(e["score"]), 3)
        })
    return results

# ------------------- Post-Processing -------------------
def clean_and_merge(results):
    seen = set()
    final = []
    for r in results:
        txt = r["text"].strip()
        txt = re.sub(r"[^\w\s\-/]", "", txt)  # clean stray chars
        key = (r["label"], txt.lower())
        if not txt or key in seen:
            continue
        seen.add(key)
        final.append({"label": r["label"], "text": txt, "confidence": r["confidence"]})
    return final

# ------------------- UI -------------------
st.title("🔍 Robust PII Extractor for FIR / Legal Docs")

sample_text = """FIR No: 0523, Date: 19/11/2017, Time: 21:33,
Police Station: Bhosari, District: Pune,
Phone: 7720010466, Name: Anand Jes,
Location: Pune City, Maharashtra, India"""

text_input = st.text_area("Paste FIR / Document text here:", sample_text, height=300)

if st.button("Extract PII"):
    with st.spinner("Extracting PII..."):
        regex_res = regex_extract(text_input)
        ner_res = ner_extract(text_input)
        merged = clean_and_merge(regex_res + ner_res)

    if merged:
        st.subheader("📑 Extracted PII")
        for r in merged:
            st.json(r)
    else:
        st.warning("No PII found in the text!")
