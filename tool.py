# tool.py
import streamlit as st
import pandas as pd
import re
from transformers import pipeline
from rapidfuzz import fuzz

st.set_page_config(page_title="PII Extractor", layout="wide")

# ------------------- Load NER Model -------------------
@st.cache_resource
def load_model():
    return pipeline("ner", model="ai4bharat/IndicNER", aggregation_strategy="simple")

ner_model = load_model()

# ------------------- Regex Patterns -------------------
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "pin": r"\b\d{6}\b"
}

# ------------------- Helpers -------------------
def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group(), "confidence": 0.95})
    return out

def ner_extract(text):
    results = []
    words = text.split()
    chunk_size = 200
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        ents = ner_model(chunk)
        for e in ents:
            results.append({
                "label": e["entity_group"],
                "text": e["word"],
                "confidence": float(e["score"])
            })
    return results

def merge_results(results):
    final = []
    for r in results:
        if not any(fuzz.ratio(r["text"].lower(), f["text"].lower()) > 90 and r["label"] == f["label"] for f in final):
            final.append(r)
    return final

def extract_pii(text: str):
    regex_hits = regex_extract(text)
    ner_hits = ner_extract(text)
    return merge_results(regex_hits + ner_hits)

# ------------------- Streamlit UI -------------------
st.title("🔍 Multilingual PII Extractor (Legal FIR Data)")

opt = st.radio("Choose Input:", ["Paste Text", "Upload CSV"])

if opt == "Paste Text":
    user_text = st.text_area("Paste FIR / Legal text:", height=200)
    if st.button("Extract PII"):
        if user_text.strip():
            with st.spinner("Extracting..."):
                results = extract_pii(user_text)
            st.json(results)
        else:
            st.warning("Please enter some text.")

else:
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("📄 Data Preview:", df.head())

        col = st.selectbox("Select column containing text:", df.columns)
        if st.button("Extract PII from CSV"):
            with st.spinner("Processing all rows... this may take time."):
                df["PII"] = df[col].astype(str).apply(lambda x: extract_pii(x))
            st.success("Done ✅")
            st.dataframe(df[["PII"]].head())
            st.download_button("Download Results", df.to_csv(index=False).encode("utf-8"), "pii_extracted.csv", "text/csv")
