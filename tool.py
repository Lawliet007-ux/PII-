import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import re
import io
import json
import stanza

# ---------------------------
# Improved Text Extraction
# ---------------------------
def extract_text_from_pdf(file_bytes):
    text = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        blocks = page.get_text("blocks")  # better layout
        page_text = "\n".join([b[4] for b in blocks if isinstance(b[4], str)])
        text.append(page_text)
    final_text = "\n".join(text)
    if not final_text.strip():
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                text.append(t)
        final_text = "\n".join(text)
    return clean_text(final_text)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------
# Schema Extraction
# ---------------------------
def extract_schema(text):
    schema = {
        "fir_no": None,
        "year": None,
        "state_name": None,
        "dist_name": None,
        "police_station": None,
        "under_acts": [],
        "under_sections": [],
        "revised_case_category": None,
        "oparty": [],
        "jurisdiction": None,
        "jurisdiction_type": None
    }

    # FIR number & Year
    m = re.search(r"FIR\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\/\-]+)", text, re.IGNORECASE)
    if m: schema["fir_no"] = m.group(1)
    m = re.search(r"Year[:\-]?\s*(\d{4})", text, re.IGNORECASE)
    if m: schema["year"] = m.group(1)

    # State, District, Police Station
    m = re.search(r"State[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if m: schema["state_name"] = m.group(1).strip()
    m = re.search(r"District[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if m: schema["dist_name"] = m.group(1).strip()
    m = re.search(r"Police\s*Station[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if m: schema["police_station"] = m.group(1).strip()

    # Acts and Sections
    acts = re.findall(r"IPC\s*\d{4}|[A-Z]+\s*\d{4}", text)
    if acts: schema["under_acts"] = list(set(acts))
    sections = re.findall(r"u/s\.?\s*([\d, ]+)", text, re.IGNORECASE)
    if sections: schema["under_sections"] = list(set([s.strip() for s in ",".join(sections).split(",")]))

    # Complainant / Accused (basic heuristic)
    comp = re.findall(r"Complainant[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    for c in comp:
        schema["oparty"].append({"role": "complainant", "name": c.strip(), "address": None})
    acc = re.findall(r"Accused[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    for a in acc:
        schema["oparty"].append({"role": "accused", "name": a.strip(), "address": None})

    # Jurisdiction (simple rule)
    m = re.search(r"Jurisdiction[:\-]?\s*([A-Za-z\s]+)", text, re.IGNORECASE)
    if m: schema["jurisdiction"] = m.group(1).strip()
    if "India" in text or "भारत" in text:
        schema["jurisdiction_type"] = "PAN_INDIA"

    return schema

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📄 FIR / Legal Document Structured PII Extractor")

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"])
if uploaded:
    file_bytes = uploaded.read()
    text = extract_text_from_pdf(file_bytes)

    st.subheader("Extracted Text (Preview)")
    st.text_area("Text", text[:3000], height=300)

    st.subheader("Extracted Structured Schema")
    schema = extract_schema(text)
    st.json(schema)

    st.download_button("⬇️ Download JSON", data=json.dumps(schema, indent=2, ensure_ascii=False),
                       file_name="fir_schema.json", mime="application/json")
