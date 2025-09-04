# app.py
"""
FIR PII Extractor (Streamlit)
- Upload an FIR PDF (Hindi/English/other Indic languages)
- Hybrid text extraction: PDF text layer via PyMuPDF; OCR fallback via Tesseract (eng+hin)
- Advanced PII extraction using a local instruction-following model (FLAN-T5)
- No regex: relies on ML prompts + light heuristic merging only

Setup notes:
1) Install Tesseract on your system and the Hindi pack (Ubuntu example):
   sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-hin

2) Python deps are listed in requirements.txt (see provided file).

3) GPU is optional. If you have CUDA, you can toggle GPU in the UI.

4) This app avoids regex intentionally, using language models for extraction.

"""

import io
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="FIR PII Extractor (OCR + LLM)",
    page_icon="🧾",
    layout="wide",
)

# -------------------------------
# UI Helpers
# -------------------------------

def tag(text: str, kind: str = "info") -> None:
    """Small colored pill-like tag."""
    colors = {
        "info": "#2563eb",
        "ok": "#16a34a",
        "warn": "#ca8a04",
        "err": "#dc2626",
    }
    st.markdown(
        f"<span style='display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:{colors.get(kind, '#2563eb')}; color:white'>{text}</span>",
        unsafe_allow_html=True,
    )

# -------------------------------
# PDF → Text (Hybrid)
# -------------------------------

def _ocr_image_pil(img: Image.Image, lang: str) -> str:
    # Light pre-processing: convert to grayscale; let Tesseract handle the rest.
    if img.mode != "L":
        img = img.convert("L")
    return pytesseract.image_to_string(img, lang=lang, config="--oem 1 --psm 6")


def _page_needs_ocr(page: fitz.Page, min_words: int = 20) -> bool:
    """Heuristic: if text layer is very sparse or empty, prefer OCR."""
    try:
        words = page.get_text("words")  # list of word tuples
        if words is None:
            return True
        return len(words) < min_words
    except Exception:
        return True


def extract_text_from_pdf(file_bytes: bytes, dpi: int = 300, ocr_lang: str = "eng+hin") -> Tuple[str, List[str]]:
    """Robust text extraction from a PDF.
    Returns (full_text, per_page_texts).
    Strategy:
      1) Try the live text layer via PyMuPDF.
      2) If page looks scanned/empty, run OCR via Tesseract on a high-DPI render.
    """
    per_page = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        # If text seems insufficient, OCR fallback
        if _page_needs_ocr(page) or len(text.strip()) < 50:
            # Render at high DPI for better OCR
            pm = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            try:
                text = _ocr_image_pil(img, ocr_lang)
            except Exception as e:
                text = text or ""
        per_page.append(text)
    full_text = "\n\n".join(per_page)
    return full_text, per_page

# -------------------------------
# LLM Loader (FLAN-T5)
# -------------------------------

@st.cache_resource(show_spinner=False)
def load_llm(model_name: str, use_gpu: bool):
    device = 0 if (use_gpu and torch.cuda.is_available()) else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return pipe

# -------------------------------
# Chunking + Prompting
# -------------------------------

SCHEMA_KEYS = [
    "fir_no",
    "year",
    "state_name",
    "dist_name",
    "police_station",
    "under_acts",
    "under_sections",
    "revised_case_category",
    "oparty",
    "name",
    "address",
    "jurisdiction",
    "jurisdiction_type",
]

EMPTY_RECORD = {
    "fir_no": "",
    "year": "",
    "state_name": "",
    "dist_name": "",
    "police_station": "",
    "under_acts": [],
    "under_sections": [],
    "revised_case_category": "",
    "oparty": [],
    "name": [],
    "address": [],
    "jurisdiction": "",
    "jurisdiction_type": "",
}


def normalize_spaces(s: str) -> str:
    return " ".join((s or "").split())


def split_into_chunks(text: str, max_chars: int = 3500) -> List[str]:
    """Split text into semi-natural chunks without regex."""
    if len(text) <= max_chars:
        return [text]
    # Prefer paragraph breaks, then lines; keep delimiters by re-joining later.
    paragraphs = text.split("\n\n")
    chunks, cur = [], []
    cur_len = 0
    for para in paragraphs:
        p = para.strip()
        if not p:
            continue
        if cur_len + len(p) + 2 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 2
        else:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


INSTRUCTIONS = (
    "You are an expert at extracting structured information from Indian FIR documents. "
    "Given the FIR text (Hindi/English may be mixed), extract the following fields and return strictly valid JSON. "
    "If a field is missing, use an empty string for single values or an empty list for lists. "
    "Do not add keys beyond the schema. Do not include any explanations. "
)

SCHEMA_JSON = json.dumps(EMPTY_RECORD, ensure_ascii=False)


def build_prompt(chunk_text: str) -> str:
    header = INSTRUCTIONS + "\n\nSchema (keys only, values are examples/placeholders):\n" + SCHEMA_JSON
    task = (
        "\n\nNow extract the data from the following FIR text and respond with JSON only:\n\n" + chunk_text
    )
    return header + task


# -------------------------------
# JSON parsing & aggregation
# -------------------------------

def try_parse_json(s: str) -> Dict[str, Any]:
    # Robust: find the first '{' and last '}' and parse that slice
    if not s:
        return EMPTY_RECORD.copy()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
        except Exception:
            obj = None
    else:
        obj = None
    if not isinstance(obj, dict):
        return EMPTY_RECORD.copy()
    # Ensure schema keys
    out = {}
    for k in SCHEMA_KEYS:
        v = obj.get(k, EMPTY_RECORD[k])
        # Normalize simple types
        if isinstance(EMPTY_RECORD[k], list):
            if isinstance(v, list):
                out[k] = [normalize_spaces(str(x)) for x in v if str(x).strip()]
            elif isinstance(v, str) and v.strip():
                out[k] = [normalize_spaces(v)]
            else:
                out[k] = []
        else:
            out[k] = normalize_spaces(str(v)) if v is not None else ""
    return out


def aggregate_records(records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Combine chunk-wise records into a single best record, plus per-field confidence.
    Confidence is a simple agreement ratio across non-empty votes.
    """
    if not records:
        return EMPTY_RECORD.copy(), {k: 0.0 for k in SCHEMA_KEYS}

    final = EMPTY_RECORD.copy()
    conf: Dict[str, float] = {}

    for k in SCHEMA_KEYS:
        vals = []
        if isinstance(EMPTY_RECORD[k], list):
            merged: List[str] = []
            for r in records:
                v = r.get(k, [])
                if isinstance(v, list):
                    merged.extend([x for x in v if x])
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for x in merged:
                key = x.lower()
                if key not in seen:
                    unique.append(x)
                    seen.add(key)
            final[k] = unique
            # Confidence: proportion of chunks that produced non-empty list
            non_empty = sum(1 for r in records if r.get(k))
            conf[k] = non_empty / len(records)
        else:
            for r in records:
                v = r.get(k, "")
                if isinstance(v, str) and v:
                    vals.append(v)
            if not vals:
                final[k] = ""
                conf[k] = 0.0
            else:
                # Choose the mode (most frequent) among non-empty values
                counts = Counter(vals)
                best, best_n = counts.most_common(1)[0]
                final[k] = best
                conf[k] = best_n / len(records)
    return final, conf


# -------------------------------
# Streamlit App
# -------------------------------

st.title("🧾 FIR PII Extractor — OCR + LLM (No Regex)")
st.caption(
    "Upload an FIR PDF (Hindi/English). The app uses hybrid OCR and a local instruction-following model to extract structured PII."
)

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "LLM model",
        [
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=0,
        help="Larger models may be more accurate but need more RAM/GPU.",
    )
    use_gpu = st.toggle("Use GPU (if available)", value=False)
    ocr_lang = st.text_input(
        "OCR language codes (Tesseract)",
        value="eng+hin",
        help="Install the matching Tesseract language packs. e.g., eng+hin",
    )
    dpi = st.slider("OCR render DPI", 200, 400, 300, step=25)
    max_chars = st.slider("Chunk size (chars)", 1500, 6000, 3500, step=250)

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    file_bytes = uploaded.read()
    with st.spinner("Extracting text from PDF…"):
        full_text, per_page = extract_text_from_pdf(file_bytes, dpi=dpi, ocr_lang=ocr_lang)

    st.success("Text extracted.")
    st.write(f"Pages detected: {len(per_page)}")

    with st.expander("Preview extracted text"):
        st.text_area("Full text", value=full_text[:20000], height=300)

    if st.button("Extract PII", type="primary"):
        pipe = load_llm(model_choice, use_gpu)
        chunks = split_into_chunks(full_text, max_chars=max_chars)

        st.write(f"Processing {len(chunks)} chunk(s) with {model_choice}…")
        records: List[Dict[str, Any]] = []
        progress = st.progress(0)
        for idx, ch in enumerate(chunks, start=1):
            prompt = build_prompt(ch)
            out = pipe(prompt, max_new_tokens=400, do_sample=False)
            text_out = out[0]["generated_text"] if out and isinstance(out, list) else ""
            rec = try_parse_json(text_out)
            records.append(rec)
            progress.progress(idx / len(chunks))

        final_rec, confidences = aggregate_records(records)

        st.subheader("Extracted Fields")
        col1, col2 = st.columns(2)
        with col1:
            st.json(final_rec)
        with col2:
            df = pd.DataFrame({"field": list(confidences.keys()), "confidence": list(confidences.values())})
            st.dataframe(df, use_container_width=True)

        # Download JSON
        out_json = json.dumps(final_rec, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download JSON",
            data=out_json.encode("utf-8"),
            file_name="fir_pii.json",
            mime="application/json",
        )

        # Simple flat table preview
        kv = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in final_rec.items()}
        st.divider()
        st.write("Preview table")
        st.table(pd.DataFrame([kv]))

else:
    tag("No file uploaded yet", "warn")
    st.info(
        "This tool avoids regex and uses ML (OCR + instruction-following LLM) to extract FIR fields."
    )
