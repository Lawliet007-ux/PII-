import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import easyocr
from PIL import Image
import numpy as np
import layoutparser as lp
from transformers import pipeline

# ---------------------------
# Initialize OCR + LLM
# ---------------------------
reader = easyocr.Reader(['en', 'hi'])  # Hindi + English OCR
normalizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small"
)

# ---------------------------
# OCR helper
# ---------------------------
def ocr_image(pil_img):
    arr = np.array(pil_img)
    results = reader.readtext(arr, detail=0, paragraph=True)
    return " ".join(results)

# ---------------------------
# PDF text extraction
# ---------------------------
def extract_text_from_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    extracted_text = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            extracted_text.append(text)
        else:
            # fallback to OCR
            pil_images = convert_from_bytes(pdf_bytes, dpi=300, first_page=page.number+1, last_page=page.number+1)
            ocr_texts = [ocr_image(img) for img in pil_images]
            extracted_text.extend(ocr_texts)

    return "\n".join(extracted_text)

# ---------------------------
# LayoutParser: get metadata region
# ---------------------------
def extract_metadata_block(text):
    # naive split: first 800 chars usually contain FIR meta info
    return text[:800]

# ---------------------------
# Normalize fields with LLM
# ---------------------------
def extract_pii_fields(text):
    instruction = """Extract FIR metadata into JSON with these fields:
    fir_no, year, state_name, dist_name, police_station,
    under_acts, under_sections, revised_case_category,
    oparty, name, address, phone, jurisdiction, jurisdiction_type."""

    input_text = f"{instruction}\n\nFIR TEXT:\n{text}"
    output = normalizer(input_text, max_length=512)[0]["generated_text"]
    return output

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📄 FIR PII Extractor (Hindi + English)")

uploaded_file = st.file_uploader("Upload FIR PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("📑 Extracted Text")
    st.text_area("Raw Text", text, height=300)

    with st.spinner("Extracting PII fields..."):
        meta_block = extract_metadata_block(text)
        pii_json = extract_pii_fields(meta_block)

    st.subheader("🧾 Extracted PII (JSON)")
    st.json(pii_json)
