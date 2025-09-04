import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import easyocr
from PIL import Image
import numpy as np
import layoutparser as lp
from transformers import pipeline
import json
import re
import json5

# ---------------------------
# Initialize OCR + LLM
# ---------------------------
@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en', 'hi'])  # Hindi + English OCR
    normalizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small"
    )
    return reader, normalizer

reader, normalizer = load_models()

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
            pil_images = convert_from_bytes(
                pdf_bytes, dpi=300,
                first_page=page.number+1, last_page=page.number+1
            )
            ocr_texts = [ocr_image(img) for img in pil_images]
            extracted_text.extend(ocr_texts)

    return "\n".join(extracted_text)

# ---------------------------
# LayoutParser: metadata block
# ---------------------------
def extract_metadata_block(text):
    # In practice you’d use lp.Detectron2LayoutModel,
    # but here fallback to first ~800 chars (metadata zone).
    return text[:800]

# ---------------------------
# JSON repair helper
# ---------------------------
def clean_and_parse_json(raw_text: str):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    candidate = match.group(0) if match else raw_text

    # Basic cleanup
    candidate = candidate.replace("):", ":")
    candidate = candidate.replace("(cid:", "")
    candidate = re.sub(r"\)\s*", "", candidate)

    try:
        return json.loads(candidate)
    except Exception:
        try:
            return json5.loads(candidate)
        except Exception:
            return {"error": "Could not parse JSON", "raw": raw_text}

# ---------------------------
# Normalize fields with LLM
# ---------------------------
def extract_pii_fields(text):
    instruction = """Extract FIR metadata into strict JSON with these fields:
    fir_no, year, state_name, dist_name, police_station,
    under_acts, under_sections, revised_case_category,
    oparty, name, address, phone, jurisdiction, jurisdiction_type.
    Ensure valid JSON format ONLY, no extra commentary."""

    input_text = f"{instruction}\n\nFIR TEXT:\n{text}"
    output = normalizer(input_text, max_length=512)[0]["generated_text"]
    return clean_and_parse_json(output)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📄 FIR PII Extractor (Hindi + English, Offline)")

uploaded_file = st.file_uploader("Upload FIR PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    st.subheader("📑 Extracted Text")
    st.text_area("Raw Text", text, height=300)

    with st.spinner("Extracting PII fields..."):
        meta_block = extract_metadata_block(text)
        pii_json = extract_pii_fields(meta_block)

    st.subheader("🧾 Extracted PII (Editable)")
    edited = {}
    if isinstance(pii_json, dict):
        for key, value in pii_json.items():
            if isinstance(value, list):
                edited[key] = st.text_area(f"{key}", "\n".join(map(str, value)))
                edited[key] = [v.strip() for v in edited[key].split("\n") if v.strip()]
            else:
                edited[key] = st.text_input(f"{key}", str(value))
    else:
        st.warning("Could not parse structured JSON. Showing raw output.")
        st.text_area("Raw Output", str(pii_json), height=200)

    if st.button("💾 Download JSON"):
        fixed_json = json.dumps(edited, ensure_ascii=False, indent=2)
        st.download_button("Download File", fixed_json, file_name="fir_pii.json", mime="application/json")
