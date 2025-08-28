# better_pii_extractor.py
import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ------------------- Setup -------------------
st.set_page_config(page_title="Better PII Extractor", layout="wide")

MODEL_NAME = "dslim/bert-base-NER"   # try "ai4bharat/IndicNER" for Indian context

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
    "time": r"\b\d{1,2}[:\.]\d{2}\b",
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

# ------------------- NER Extract + Join -------------------
def ner_extract(text):
    ents = nlp(text)
    merged = []
    buffer = []
    prev_label = None

    for e in ents:
        word = e["word"].strip()
        if e["score"] < 0.6 or len(word) < 2 or word.startswith("##"):
            continue

        label = e["entity_group"].lower()

        if prev_label == label:   # same entity → join
            buffer.append(word)
        else:
            if buffer:
                merged.append({"label": prev_label, "text": " ".join(buffer), "confidence": round(float(e["score"]), 3)})
            buffer = [word]
            prev_label = label

    if buffer:
        merged.append({"label": prev_label, "text": " ".join(buffer), "confidence": 0.9})

    return merged

# ------------------- Post-Processing -------------------
def clean_and_merge(results):
    seen = set()
    final = []
    for r in results:
        txt = r["text"].strip()
        txt = re.sub(r"[^\w\s\-/]", "", txt)
        if len(txt) < 2:  # drop junk
            continue
        key = (r["label"], txt.lower())
        if key in seen:
            continue
        seen.add(key)
        final.append({"label": r["label"], "text": txt, "confidence": r["confidence"]})
    return final

# ------------------- UI -------------------
st.title("🔍 Better PII Extractor (Legal / FIR Docs)")

sample_text = """FIR No: 0523, Date: 19/11/2017, Time: 21:33,
Police Station: Bhosari, District: Pune,
Phone: 7720010466, Name: Anand Jes,
Complainant: Rev. Dinanath, Accused: Sanjay Kumar,
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
