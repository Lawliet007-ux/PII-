"""
Streamlit FIR PII Extractor (single-file)

Features:
- Upload FIR PDF (scanned or digital)
- Robust text extraction: pdfplumber -> PyMuPDF (fitz) -> OCR (pdf2image + pytesseract)
- Multilingual support (Hindi + English). Choose OCR langs (hin, eng).
- PII extraction using advanced techniques:
  - Multilingual NER (Hugging Face XLM-R models)
  - Extractive Question Answering (multilingual XLM-R QA) for fields not covered by NER
  - Phone detection using `phonenumbers` library (no regex)
- Outputs JSON with confidence scores and a downloadable file

Notes / requirements (install on your machine):
- System: install Tesseract OCR and Hindi traineddata (e.g. `tesseract-ocr`, `tesseract-ocr-hin` on Debian/Ubuntu)
- Python packages (example):
    pip install streamlit pdfplumber pymupdf pdf2image pillow pytesseract transformers torch phonenumbers langdetect sentencepiece

Run with:
    streamlit run streamlit_fir_pii_extractor.py

"""

import io
import json
import math
from typing import List, Dict, Any

import streamlit as st

# PDF / OCR
import pdfplumber
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# NLP / Transformers
import torch
from transformers import pipeline

# Utilities
import phonenumbers
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# --- Streamlit UI ---
st.set_page_config(page_title="FIR PII Extractor (Multilingual)", layout="wide")
st.title("FIR PII Extractor — Multilingual (Hindi + English)")
st.markdown("Upload an FIR PDF (scanned or digital). The app will try multiple extraction strategies and then use NER + extractive QA to recover structured PII fields.")

# Sidebar settings
st.sidebar.header("Settings")
ocr_langs = st.sidebar.multiselect("OCR languages (Tesseract)", options=["eng", "hin"], default=["eng", "hin"])
use_ocr_always = st.sidebar.checkbox("Always run OCR (even for text PDFs)", value=False)
ner_model_choice = st.sidebar.selectbox("NER model", options=["Davlan/xlm-roberta-base-wikiann-ner", "Venkatesh4342/NER-Indian-xlm-roberta"], index=0)
qa_model_choice = st.sidebar.selectbox("QA model", options=["deepset/xlm-roberta-base-squad2", "deepset/xlm-roberta-large-squad2"], index=0)
qa_score_threshold = st.sidebar.slider("QA confidence threshold", 0.0, 1.0, 0.25)

st.sidebar.markdown("---")
st.sidebar.markdown("System & model notes:\n- Tesseract must be installed on the host for OCR to work.\n- Models will be downloaded from Hugging Face on first run (requires internet).\n- Large models may require significant RAM/VRAM.")

uploaded_file = st.file_uploader("Upload FIR PDF", type=["pdf"], accept_multiple_files=False)

# --- Helpers ---

@st.cache_resource(ttl=3600)
def load_ner_pipeline(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple", device=device)
    return ner

@st.cache_resource(ttl=3600)
def load_qa_pipeline(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    qa = pipeline("question-answering", model=model_name, tokenizer=model_name, device=device)
    return qa


def extract_text_pdf_bytes(pdf_bytes: bytes, run_ocr_always: bool = False, ocr_langs: List[str] = ["eng", "hin"]) -> str:
    """Try: pdfplumber -> PyMuPDF -> OCR (pdf2image + pytesseract).
    Return the best (longest) extracted text string.
    """
    results = []
    # attempt pdfplumber text extraction
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(text_pages).strip()
            if text:
                results.append(("pdfplumber", text))
    except Exception as e:
        st.debug = getattr(st, "debug", lambda *a, **k: None)
        st.debug(f"pdfplumber fail: {e}")

    # attempt PyMuPDF (fitz)
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_pages = []
        for p in doc:
            try:
                t = p.get_text("text")
            except Exception:
                t = ""
            text_pages.append(t or "")
        text = "\n".join(text_pages).strip()
        if text:
            results.append(("pymupdf", text))
    except Exception as e:
        st.debug(f"pymupdf fail: {e}")

    # decide if we need OCR
    longest_text = max([t for (_, t) in results], key=len) if results else ""
    need_ocr = run_ocr_always or (len(longest_text) < 200)

    if need_ocr:
        # convert PDF pages to images and run Tesseract
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300)
            ocr_text_pages = []
            tesseract_lang = "+".join(ocr_langs) if ocr_langs else None
            tesseract_config = r"--psm 6"  # assume a block of text
            for img in images:
                # preprocess image (optional): convert to grayscale, increase contrast etc.
                gray = img.convert("L")
                ocr_res = pytesseract.image_to_string(gray, lang=tesseract_lang, config=tesseract_config)
                ocr_text_pages.append(ocr_res or "")
            ocr_text = "\n".join(ocr_text_pages).strip()
            if ocr_text:
                results.append(("ocr", ocr_text))
        except Exception as e:
            st.debug(f"OCR fail: {e}")

    # return the longest candidate
    if not results:
        return ""
    best = max(results, key=lambda x: len(x[1]))
    source, text = best
    st.info(f"Using text from: {source} (length={len(text)} chars)")
    return text


def chunk_text(text: str, max_chunk_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Split long text into overlapping chunks for QA models.
    This is character-based splitting to be simple and robust.
    """
    if len(text) <= max_chunk_chars:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_chars, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks


def extract_field_via_qa(qa_pipeline, text: str, question: str, top_k: int = 3) -> Dict[str, Any]:
    """Run QA over chunks and return best answer with score.
    """
    chunks = chunk_text(text)
    best = {"answer": "", "score": 0.0, "context": ""}
    for c in chunks:
        try:
            res = qa_pipeline(question=question, context=c)
            # If model returns dict, use directly; if list, iterate
            if isinstance(res, list):
                for r in res:
                    if r.get("score", 0) > best["score"]:
                        best = {"answer": r.get("answer"), "score": r.get("score", 0.0), "context": c}
            else:
                if res.get("score", 0) > best["score"]:
                    best = {"answer": res.get("answer"), "score": res.get("score", 0.0), "context": c}
        except Exception as e:
            st.debug(f"QA chunk failed: {e}")
    return best


def find_phone_numbers(text: str) -> List[str]:
    numbers = []
    try:
        for m in phonenumbers.PhoneNumberMatcher(text, "IN"):
            num = phonenumbers.format_number(m.number, phonenumbers.PhoneNumberFormat.E164)
            numbers.append(num)
    except Exception as e:
        st.debug(f"phone parse fail: {e}")
    return list(dict.fromkeys(numbers))

# --- Main flow ---

if uploaded_file is None:
    st.info("Upload a PDF to start.")
    st.stop()

pdf_bytes = uploaded_file.read()

with st.spinner("Extracting text from PDF (this may take a while)..."):
    extracted_text = extract_text_pdf_bytes(pdf_bytes, run_ocr_always=use_ocr_always, ocr_langs=ocr_langs)

if not extracted_text:
    st.error("Failed to extract text from the PDF. Make sure Tesseract is installed for OCR or try a higher DPI.")
    st.stop()

# show a preview
st.subheader("Extracted text (preview)")
with st.expander("Show full extracted text"):
    st.text_area("Extracted text", extracted_text, height=400)

# language detection
try:
    detected_lang = detect(extracted_text)
except Exception:
    detected_lang = "unknown"
st.markdown(f"**Detected language (rough):** {detected_lang}")

# load models
with st.spinner("Loading NER & QA models (cached)..."):
    ner_pipeline = load_ner_pipeline(ner_model_choice)
    qa_pipeline = load_qa_pipeline(qa_model_choice)

# Run NER
st.subheader("Named Entity Recognition (NER)")
with st.spinner("Running NER..."):
    try:
        ner_results = ner_pipeline(extracted_text)
    except Exception as e:
        st.error(f"NER pipeline failed: {e}")
        ner_results = []

# Format NER results
entities_by_type = {}
for ent in ner_results:
    label = ent.get("entity_group") or ent.get("entity")
    entities_by_type.setdefault(label, []).append({"text": ent.get("word"), "score": ent.get("score")})

st.write(entities_by_type)

# Use QA to extract structured FIR fields
st.subheader("Structured field extraction (QA + heuristics)")
fields = {
    "fir_no": "What is the FIR number?",
    "year": "What is the year of the FIR?",
    "state_name": "Which state is mentioned in the FIR?",
    "dist_name": "Which district is mentioned?",
    "police_station": "Which police station is mentioned in the FIR?",
    "under_acts": "Under which Acts is the case registered?",
    "under_sections": "Which sections are mentioned under the FIR?",
    "revised_case_category": "Which case category is mentioned?",
    "oparty": "Who is the opposite party / complainant / accused?",
    "names': 'List the names of people mentioned in the FIR.'": "Who are the people mentioned?",
    "address": "What address is mentioned in the FIR?",
    "jurisdiction": "What jurisdiction is mentioned?",
}

extracted_structured = {}
for key, q in fields.items():
    with st.spinner(f"Extracting {key} ..."):
        try:
            ans = extract_field_via_qa(qa_pipeline, extracted_text, q)
            if ans and ans.get("score", 0) >= qa_score_threshold:
                extracted_structured[key] = {"value": ans.get("answer"), "score": ans.get("score")}
            else:
                extracted_structured[key] = {"value": "", "score": ans.get("score", 0.0)}
        except Exception as e:
            st.debug(f"Extraction {key} failed: {e}")
            extracted_structured[key] = {"value": "", "score": 0.0}

# Names and persons from NER
persons = []
for label, items in entities_by_type.items():
    if label and label.upper().startswith("PER") or label.upper() == "PERSON":
        for it in items:
            persons.append(it["text"])

extracted_structured["names"] = {"value": list(dict.fromkeys(persons)), "source": "ner"}

# Phones
phones = find_phone_numbers(extracted_text)
extracted_structured["phones"] = {"value": phones, "source": "phonenumbers"}

# Compose final JSON
final_output = {
    "metadata": {
        "source_filename": uploaded_file.name,
        "text_length": len(extracted_text),
    },
    "extracted_fields": extracted_structured,
}

st.subheader("Final JSON")
st.json(final_output)

# Download
st.download_button("Download JSON", data=json.dumps(final_output, ensure_ascii=False, indent=2), file_name="fir_extraction.json", mime="application/json")

st.success("Extraction complete. Review results and adjust settings if needed.")

# End of file
