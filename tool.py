# tool.py
import streamlit as st
import re
import spacy

# Load spaCy model for NER (English + works partially with mixed language)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="PII Extractor from Elasticsearch Text", layout="wide")

st.title("🔍 PII Extractor from Legal/Elasticsearch Text")

# ------------------- Helper Functions -------------------

def extract_regex_pii(text: str):
    """Extracts PII using regex rules for FIR, dates, times, IDs etc."""
    pii = {}

    # FIR Numbers
    fir_numbers = re.findall(r"FIR\s*No\.?\s*[:\-]?\s*([\w\/\-]+)", text, re.IGNORECASE)
    if fir_numbers:
        pii["FIR_Number"] = list(set(fir_numbers))

    # Dates (dd/mm/yyyy or dd-mm-yyyy)
    dates = re.findall(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", text)
    if dates:
        pii["Dates"] = list(set(dates))

    # Times (hh:mm format)
    times = re.findall(r"\b(\d{1,2}:\d{2})\s*(?:तास|hours|hrs|h)?", text)
    if times:
        pii["Times"] = list(set(times))

    # Entry/Diary numbers
    entry_numbers = re.findall(r"Entry\s*No\.?\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if entry_numbers:
        pii["Entry_Numbers"] = list(set(entry_numbers))

    # Phone Numbers (10 digit Indian)
    phones = re.findall(r"\b\d{10}\b", text)
    if phones:
        pii["Phone_Numbers"] = list(set(phones))

    return pii


def extract_ner_pii(text: str):
    """Extracts PII using spaCy NER"""
    doc = nlp(text)
    pii = {
        "PERSON": [],
        "ORG": [],
        "GPE": [],  # Geo-political entities
        "LOC": []
    }

    for ent in doc.ents:
        if ent.label_ in pii:
            pii[ent.label_].append(ent.text)

    # Deduplicate
    for k in pii:
        pii[k] = list(set(pii[k]))

    return pii


def merge_dicts(d1, d2):
    """Merge two PII dicts"""
    merged = d1.copy()
    for k, v in d2.items():
        if k in merged:
            merged[k].extend(v)
            merged[k] = list(set(merged[k]))
        else:
            merged[k] = v
    return merged

# ------------------- Streamlit UI -------------------

sample_text = """P.S. (पोलीस ठाणे): भोसरी
FIR No. (Ĥम खब Đ.): 0523
Date and Time of FIR: 19/11/2017 21:33 वाजता
District: पुणे शहर
Complainant Name: VIPUL RANGNATH JADHAV
Father's Name: RANGNATH JADHAV
Address: मोया माळाात, ाशिाटा, ासारवाडि, पुणे
"""

text_input = st.text_area("Paste Elasticsearch/Legal Text here:", sample_text, height=400)

if st.button("Extract PII"):
    regex_pii = extract_regex_pii(text_input)
    ner_pii = extract_ner_pii(text_input)
    final_pii = merge_dicts(regex_pii, ner_pii)

    st.subheader("📌 Extracted PII")
    st.json(final_pii)
