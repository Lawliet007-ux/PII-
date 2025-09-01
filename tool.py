"""
Streamlit app: FIR PII extractor (Hindi + English)
Enhanced version: works with uploaded PDFs and pasted text.
"""

import streamlit as st
import re
import json
import base64
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import tempfile
import os

st.set_page_config(page_title="FIR PII Extractor", layout="wide")

# ------------------ Text Extraction ------------------

def extract_text_from_pdf(path):
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except:
        pass
    if not text.strip():
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        text += t + "\n"
        except:
            pass
    if not text.strip():
        try:
            doc = fitz.open(path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img, lang="eng+hin") + "\n"
        except:
            pass
    return text

# ------------------ PII Extraction ------------------

def extract_pii_rules(text):
    result = {}
    clean_text = re.sub(r"[ \t]+", " ", text)

    # Year
    m = re.search(r"(19|20)\d{2}", clean_text)
    result["year"] = m.group(0) if m else None

    # State
    if "महारा" in clean_text or "Maharashtra" in clean_text:
        result["state_name"] = "Maharashtra"
    elif "दिल्ली" in clean_text or "Delhi" in clean_text:
        result["state_name"] = "Delhi"
    else:
        result["state_name"] = None

    # District
    m = re.search(r"District[:\s]+([A-Za-z ]+)", clean_text, re.IGNORECASE)
    if not m:
        m = re.search(r"जिला[:\s]+([\u0900-\u097F ]+)", clean_text)
    result["dist_name"] = m.group(1).strip() if m else None

    # Police Station
    m = re.search(r"(?:Police Station|P\.S\.|पोलीस ठाणे)[:\s]+([A-Za-z\u0900-\u097F ]+)", clean_text)
    result["police_station"] = m.group(1).strip() if m else None

    # Acts
    acts_found = re.findall(r"(Indian Penal Code|IPC|Arms Act,? ?\d{4}|Maharashtra Police Act,? ?\d{4})", clean_text, re.IGNORECASE)
    acts_found = list(dict.fromkeys([a.strip() for a in acts_found]))
    result["under_acts"] = acts_found if acts_found else None

    # Sections
    sections_found = re.findall(r"\b\d{1,3}(?:\(\d+\))?\b", clean_text)
    sections_filtered = []
    for s in sections_found:
        try:
            val = int(re.sub(r"[^0-9]", "", s))
            if 0 < val < 600:
                sections_filtered.append(s)
        except:
            pass
    sections_filtered = list(dict.fromkeys(sections_filtered))
    result["under_sections"] = sections_filtered if sections_filtered else None

    # Case category
    if result.get("under_acts") and any("Arms" in a or "हत्यार" in a for a in result["under_acts"]):
        result["revised_case_category"] = "WEAPONS"
    elif result.get("under_sections") and any(s in ["354", "376", "509"] for s in result["under_sections"]):
        result["revised_case_category"] = "CRIMINAL_OFFENCE_SEXUAL_HARASSMENT"
    else:
        result["revised_case_category"] = "OTHER"

    # Name / Opposite party
    m = re.search(r"Name[:\s]+([A-Z][A-Za-z ]+)", clean_text)
    if m:
        result["name"] = m.group(1).strip()
        result["oparty"] = "Complainant"
    else:
        result["name"] = None
        result["oparty"] = None

    # Address
    m = re.search(r"Address[:\s]+([^\n]+)", clean_text)
    if m:
        result["address"] = m.group(1).strip()
    else:
        result["address"] = None

    # Jurisdiction
    if result.get("dist_name"):
        result["jurisdiction"] = result["dist_name"]
        result["jurisdiction_type"] = "DISTRICT"
    else:
        result["jurisdiction"] = None
        result["jurisdiction_type"] = None

    return result

# ------------------ Streamlit UI ------------------

st.title("FIR PII Extractor — Hindi + English")

st.write("Upload FIR PDF(s) or paste text manually to extract structured PII.")

uploaded_files = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
raw_text = st.text_area("Or paste FIR text here", height=200)

if st.button("Extract PII"):
    results = {}
    if uploaded_files:
        for uploaded in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            extracted_text = extract_text_from_pdf(tmp_path)
            pii = extract_pii_rules(extracted_text)
            results[uploaded.name] = pii
            os.remove(tmp_path)
    if raw_text.strip():
        pii = extract_pii_rules(raw_text)
        results["pasted_text"] = pii
    if results:
        st.subheader("Extracted PII")
        st.json(results)
        b64 = base64.b64encode(json.dumps(results, ensure_ascii=False, indent=2).encode()).decode()
        href = f"data:application/json;base64,{b64}"
        st.markdown(f"[Download JSON]({href})")
    else:
        st.warning("Please upload a file or paste text.")
