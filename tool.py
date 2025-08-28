# tool.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ------------------- Setup -------------------
st.set_page_config(page_title="PII Extractor", layout="wide")

MODEL_NAME = "dslim/bert-base-NER"   # replace with your fine-tuned model if available

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp, tokenizer

nlp, tokenizer = load_model()

# ------------------- Helpers -------------------
def chunk_text(text, max_length=512, stride=256):
    """Split long text into overlapping chunks within model's limit"""
    tokens = tokenizer(
        text,
        return_overflowing_tokens=True,
        max_length=max_length,
        stride=stride,
        truncation=True
    )
    chunks = tokens["input_ids"]
    chunk_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in chunks]
    return chunk_texts

def extract_pii(text):
    """Extract PII from long text using chunking"""
    results = []
    chunks = chunk_text(text)
    for chunk in chunks:
        entities = nlp(chunk)
        for ent in entities:
            results.append({
                "label": ent["entity_group"],
                "text": ent["word"],
                "confidence": round(ent["score"], 3)
            })
    return results

# ------------------- UI -------------------
st.title("🔍 PII Extractor for FIR / Legal Docs")

sample_text = """FIR No: 0523, Date: 19/11/2017, Time: 21:33,
Police Station: Bhosari, District: Pune,
Phone: 7720010466, Name: Anand Jes, Location: Pune City, Maharashtra, India"""

text_input = st.text_area("Paste FIR / Document text here:", sample_text, height=300)

if st.button("Extract PII"):
    with st.spinner("Extracting PII..."):
        results = extract_pii(text_input)

    if results:
        st.subheader("📑 Extracted PII")
        for r in results:
            st.json(r)
    else:
        st.warning("No PII found in the text!")
