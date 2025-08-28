# tool.py
import re
import json
import unicodedata
import streamlit as st
from typing import List, Dict
import spacy

st.set_page_config(page_title="📑 FIR PII Extractor", layout="wide")

# ---------- Load spaCy model ----------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ---------- Text cleaning ----------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- Regex extractors ----------
def regex_extract(text: str) -> List[Dict]:
    out = []

    # Phone numbers
    for m in re.finditer(r"(?:\+91[-\s]?)?[6-9]\d{9}", text):
        out.append({"label": "phone", "text": m.group(), "confidence": 1.0, "origin": "regex"})

    # Dates
    for m in re.finditer(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", text):
        out.append({"label": "date", "text": m.group(), "confidence": 1.0, "origin": "regex"})

    # Times
    for m in re.finditer(r"\b(\d{1,2}[:.]\d{2})\b", text):
        out.append({"label": "time", "text": m.group().replace(".", ":"), "confidence": 1.0, "origin": "regex"})

    # FIR No
    m = re.search(r"FIR\s*No[^0-9A-Za-z]*([\w-]+)", text, re.IGNORECASE)
    if m:
        out.append({"label": "fir_no", "text": m.group(1), "confidence": 0.99, "origin": "regex"})

    # DOB
    m = re.search(r"Date\s*/?\s*Year of Birth[^0-9]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.IGNORECASE)
    if m:
        out.append({"label": "dob", "text": m.group(1), "confidence": 0.99, "origin": "regex"})

    return out

# ---------- Label-aware parsing ----------
def label_parse(text: str) -> List[Dict]:
    out = []
    lower = text.lower()

    # Police station
    m = re.search(r"(police\s*(station|thane)[^:]*:\s*([A-Za-z]+))", text, re.IGNORECASE)
    if m:
        out.append({"label": "police_station", "text": m.group(3), "confidence": 0.98, "origin": "label"})

    # District
    m = re.search(r"district[^:]*:\s*([A-Za-z ]+)", text, re.IGNORECASE)
    if m:
        out.append({"label": "district", "text": m.group(1).strip(), "confidence": 0.98, "origin": "label"})

    # Nationality
    m = re.search(r"nationality[^:]*:\s*([A-Za-z ]+)", text, re.IGNORECASE)
    if m:
        out.append({"label": "nationality", "text": m.group(1).strip(), "confidence": 0.95, "origin": "label"})

    return out

# ---------- Address block assembly ----------
def extract_addresses(text: str) -> List[Dict]:
    out = []
    addr_blocks = re.findall(r"Address[^:]*:\s*([^:]+?)(?=(?:[A-Z][a-z]+:|$))", text, re.IGNORECASE)
    for block in addr_blocks:
        cleaned = clean_text(block)
        if len(cleaned) > 10:
            out.append({"label": "address", "text": cleaned, "confidence": 0.9, "origin": "address_block"})
    return out

# ---------- spaCy NER ----------
def spacy_entities(text: str) -> List[Dict]:
    out = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            # filter out junk
            if len(ent.text.strip()) > 2 and not ent.text.isdigit():
                out.append({
                    "label": ent.label_.lower(),
                    "text": ent.text.strip(),
                    "confidence": 0.75,
                    "origin": "spacy"
                })
    return out

# ---------- Combine & deduplicate ----------
def dedup(results: List[Dict]) -> List[Dict]:
    seen = set()
    final = []
    for r in results:
        key = (r["label"], r["text"].lower())
        if key not in seen:
            seen.add(key)
            final.append(r)
    return final

# ---------- Streamlit UI ----------
st.title("📑 FIR PII Extractor")

sample_text = st.text_area("Paste FIR text here:", height=300)

if st.button("🔍 Extract PII"):
    if not sample_text.strip():
        st.warning("Please paste some text first.")
    else:
        text = clean_text(sample_text)

        results = []
        results.extend(regex_extract(text))
        results.extend(label_parse(text))
        results.extend(extract_addresses(text))
        results.extend(spacy_entities(text))

        results = dedup(results)

        st.subheader("📑 Extracted PII")
        st.json(results)
