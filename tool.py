import streamlit as st
import fitz  # PyMuPDF
import io
import json
from pdf2image import convert_from_path
import cv2
import numpy as np
from paddleocr import PaddleOCR
import layoutparser as lp
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
# OCR setup (offline)
# -----------------------
ocr = PaddleOCR(use_angle_cls=True, lang='hi')  # Hindi + English

# -----------------------
# Load local FLAN-T5 model (small for speed)
# -----------------------
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

llm_pipe = load_llm()

# -----------------------
# Helpers
# -----------------------
def preprocess_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    extracted_text = ""
    for page in doc:
        txt = page.get_text("text")
        if txt and len(txt.strip()) > 20:
            extracted_text += txt + "\n"
    # If no text, fallback OCR
    if len(extracted_text.strip()) < 50:
        images = convert_from_path(file, dpi=300)
        for img in images:
            pre = preprocess_image(img)
            result = ocr.ocr(pre, cls=True)
            for line in result[0]:
                extracted_text += line[1][0] + " "
    return extracted_text.strip()

def extract_metadata_with_llm(text):
    prompt = f"""
    Extract FIR details from the text below and return valid JSON with keys:
    fir_no, year, state_name, dist_name, police_station, under_acts, under_sections,
    revised_case_category, oparty, name, phone, address, jurisdiction, jurisdiction_type.

    Text:
    {text}
    """
    result = llm_pipe(prompt, max_length=512, clean_up_tokenization_spaces=True)
    try:
        parsed = json.loads(result[0]['generated_text'])
    except Exception:
        parsed = {"error": "LLM could not produce valid JSON", "raw_output": result[0]['generated_text']}
    return parsed

# -----------------------
# Streamlit UI
# -----------------------
st.title("FIR PII Extractor (Offline - Hindi/English)")
st.write("Upload an FIR PDF (English/Hindi). The tool extracts text → parses PII → outputs JSON.")

uploaded_file = st.file_uploader("Upload FIR PDF", type=["pdf"])

if uploaded_file:
    st.info("Processing PDF… please wait.")
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text")
    st.text_area("Raw text", text, height=300)

    with st.spinner("Extracting PII using LLM..."):
        pii_data = extract_metadata_with_llm(text)

    st.subheader("Extracted FIR Metadata")
    st.json(pii_data)

    # Download button
    st.download_button(
        "Download JSON",
        data=json.dumps(pii_data, ensure_ascii=False, indent=2),
        file_name="fir_metadata.json",
        mime="application/json"
    )
