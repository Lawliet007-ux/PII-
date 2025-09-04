# app.py
"""
FIR PII Extractor — Robust final version (offline; Hindi+English)
- Python 3.13 safe (no cv2)
- PyMuPDF text-layer -> EasyOCR fallback (per-page)
- Metadata block detection by keywords + context window
- Per-field constrained LLM extraction (Flan-T5-small locally)
- Post-processing normalization and confidence scoring
- Streamlit UI with editable fields + evidence + JSON download
"""

import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import easyocr
from PIL import Image
import numpy as np
from transformers import pipeline
import json
import re
from rapidfuzz import process, fuzz
import html
import json5
from typing import Dict, Any, List, Tuple

# -----------------------
# Config / Schema
# -----------------------
SCHEMA_KEYS = [
    "fir_no", "year", "state_name", "dist_name", "police_station",
    "under_acts", "under_sections", "revised_case_category",
    "oparty", "name", "phone", "address", "jurisdiction", "jurisdiction_type"
]

# Small gazetteer for state fuzzy matching (expand offline as needed)
INDIAN_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka",
    "Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
    "Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Delhi","Jammu and Kashmir",
    "Ladakh"
]
# Hindi variants (small set) — expand as needed
INDIAN_STATES_HINDI = [
    "महाराष्ट्र","छत्तीसगढ़","उत्तर प्रदेश","मध्य प्रदेश","बिहार","पश्चिम बंगाल","तमिलनाडु",
    "कर्नाटक","गुजरात","राजस्थान","केरल","हरियाणा","दिल्ली","ओडिशा","तेलंगाना"
]
ALL_STATES_GAZ = INDIAN_STATES + INDIAN_STATES_HINDI

# Keywords to detect metadata/header region (English + Hindi)
HEADER_KEYWORDS = [
    "FIR", "F.I.R", "First Information", "Police Station", "P.S.", "P.S", "Date of FIR", "Date",
    "Time", "Year", "District", "District (State)", "District (State)", "District (State):",
    "निरीक्षक","पोलीस","फिर","ठाणे","तारीख","वर्ष","P.S", "P.S.", "Name of P.S", "नाम", "ठाणे"
]

# -----------------------
# Caches for heavy resources
# -----------------------
@st.cache_resource(show_spinner=False)
def load_ocr_and_llm():
    reader = easyocr.Reader(['en', 'hi'])  # loads once (slow)
    # FLAN-T5 small (local)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        tokenizer="google/flan-t5-small",
        device=-1  # CPU safe; change to 0 if you have GPU
    )
    return reader, llm

reader, llm = load_ocr_and_llm()

# -----------------------
# Utility: cleaning raw OCR/text
# -----------------------
def clean_text_piece(s: str) -> str:
    if not s:
        return ""
    # remove (cid:###) tokens and stray control chars
    s = re.sub(r"\(cid:\d+\)", " ", s)
    s = s.replace("�", " ")
    s = s.replace("\r", " ")
    # remove broken parenthetical artifacts like '):' or '(. . )'
    s = re.sub(r"\)\s*:", ":", s)
    s = re.sub(r"\(\s*\.\s*\.\s*\)", " ", s)
    # unify whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------
# Extract text: prefer PDF text-layer, fallback OCR per-page
# -----------------------
def extract_full_text(pdf_bytes: bytes, ocr_dpi: int = 300) -> Tuple[str, List[str]]:
    """
    Returns (full_text, per_page_texts)
    """
    per_page_texts: List[str] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"PyMuPDF open error: {e}")

    for page in doc:
        # try text layer
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        text = clean_text_piece(text)
        if len(text) < 60:
            # fallback OCR on this page only
            try:
                images = convert_from_bytes(pdf_bytes, dpi=ocr_dpi, first_page=page.number+1, last_page=page.number+1)
                page_texts = []
                for img in images:
                    txt = reader.readtext(np.array(img), detail=0, paragraph=True)
                    if isinstance(txt, list):
                        page_texts.append(" ".join(txt))
                text = clean_text_piece(" ".join(page_texts))
            except Exception as e:
                # last resort: keep current short text
                text = clean_text_piece(text)
        per_page_texts.append(text)
    full = "\n\n".join(per_page_texts)
    full = clean_text_piece(full)
    return full, per_page_texts

# -----------------------
# Detect header/metadata block robustly
# -----------------------
def detect_metadata_block(full_text: str, per_page_texts: List[str]) -> Tuple[str, int]:
    """
    Strategy:
    - Look for header keywords on each page.
    - Return the page text and the char-window around the first match.
    - Also fallback to first page top portion.
    """
    # scan pages for header keywords
    for idx, page_text in enumerate(per_page_texts):
        lower = page_text.lower()
        for kw in HEADER_KEYWORDS:
            if kw.lower() in lower:
                # find first occurrence and take a window of ±800 chars
                pos = lower.find(kw.lower())
                start = max(0, pos - 400)
                end = min(len(page_text), pos + 400)
                snippet = clean_text_piece(page_text[start:end])
                return snippet, idx
    # fallback: first page first 1200 chars
    if per_page_texts:
        fallback = clean_text_piece(per_page_texts[0][:1200])
        return fallback, 0
    # final fallback: use beginning of full text
    return clean_text_piece(full_text[:1200]), 0

# -----------------------
# Constrained per-field LLM extractor
# -----------------------
def llm_extract_field(field_name: str, context: str) -> str:
    """
    Ask the LLM to return ONLY the field value(s) for a single field.
    This makes the model's job narrower and more reliable.
    Returns a plain string (for lists it may be comma/newline separated).
    """
    # design prompt: strict, short, example-based
    prompt = f"""You are an extractor. Given the FIR snippet, return ONLY the value for the field named '{field_name}'. 
If the field is not present, return an empty string. If there are multiple values, separate them by the pipe character ' | ' (no extra text).

FIR SNIPPET:
\"\"\"{context}\"\"\"
"""
    try:
        out = llm(prompt, max_length=256, do_sample=False)
        raw = out[0]["generated_text"]
        raw = raw.strip().strip('"').strip("'")
        raw = clean_text_piece(raw)
        return raw
    except Exception as e:
        # on failure return empty
        return ""

# -----------------------
# Post-processing / Normalization
# -----------------------
def normalize_phone(s: str) -> str:
    # keep digits and +; prefer Indian +91
    if not s:
        return ""
    digits = re.sub(r"[^\d+]", "", s)
    # if 10 digits -> prefix +91
    d = re.sub(r"[^\d]", "", digits)
    if len(d) == 10:
        return "+91-" + d
    if len(d) == 11 and d.startswith("0"):
        return "+91-" + d[1:]
    if len(d) >= 10 and len(d) <= 13:
        return "+" + d
    return digits

def normalize_year(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"(20\d{2}|19\d{2})", s)
    return m.group(0) if m else s.strip()

def normalize_sections(s: str) -> List[str]:
    if not s:
        return []
    # split on non-digit groups but try to let LLM give pipe-separated or comma-separated
    parts = re.split(r"[^\d]+", s)
    parts = [p.lstrip("0") or p for p in parts if p.strip()]
    # filter obviously wrong long numbers (>4 digits) but keep if small
    out = [p for p in parts if 1 <= len(p) <= 4]
    # dedupe preserve order
    seen = set()
    res = []
    for p in out:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res

def fuzzy_state(s: str) -> Tuple[str, float]:
    if not s:
        return "", 0.0
    match, score, _ = process.extractOne(s, ALL_STATES_GAZ, scorer=fuzz.WRatio) or ("", 0, None)
    return match or "", score/100.0

# -----------------------
# High-level orchestrator
# -----------------------
def extract_all_fields(pdf_bytes: bytes) -> Dict[str, Any]:
    full_text, pages = extract_full_text(pdf_bytes)
    meta_snip, meta_page = detect_metadata_block(full_text, pages)

    results: Dict[str, Any] = {}
    confidences: Dict[str, float] = {}
    evidence: Dict[str, str] = {}

    # For critical fields: we'll query LLM individually
    for key in SCHEMA_KEYS:
        raw_val = llm_extract_field(key, meta_snip)
        # fallback: try scanning the whole document for explicit keyword lines (cheap heuristic)
        if not raw_val:
            # quick heuristic search lines containing the field name or common patterns
            lines = [l for l in full_text.splitlines() if l.strip()]
            found = ""
            for ln in lines:
                if key.replace("_", " ").lower() in ln.lower():
                    found = ln.strip()
                    break
            if found:
                raw_val = found

        # normalize per-field
        norm = raw_val
        conf = 0.0
        if key == "phone":
            norm2 = normalize_phone(norm)
            norm = norm2
            conf = 0.9 if re.search(r"\d{10}", norm) else (0.5 if norm else 0.0)
        elif key == "year":
            norm2 = normalize_year(norm)
            norm = norm2
            conf = 0.9 if re.match(r"^(19|20)\d{2}$", norm) else (0.4 if norm else 0.0)
        elif key in ("under_sections",):
            arr = normalize_sections(norm)
            norm = arr
            conf = 0.8 if arr else 0.2
        elif key in ("under_acts",):
            # split by pipes/commas/newlines
            parts = [p.strip() for p in re.split(r"[,\n\|;/]+", norm) if p.strip()]
            norm = parts
            conf = 0.6 if parts else 0.2
        elif key == "state_name":
            match, score = fuzzy_state(norm)
            if score > 0.6:
                norm = match
                conf = score
            else:
                norm = norm
                conf = 0.3 if norm else 0.0
        else:
            # other fields: leave as string or list
            if isinstance(norm, list):
                norm = [clean_text_piece(x) for x in norm]
                conf = 0.6 if norm else 0.0
            else:
                norm = clean_text_piece(norm)
                conf = 0.7 if norm and len(norm) > 1 else (0.2 if norm else 0.0)

        results[key] = norm
        confidences[key] = round(float(conf), 2)
        # create evidence: show short snippet from meta_snip if found, else empty
        evidence[key] = meta_snip if norm else ""

    # Add metadata
    meta = {
        "meta_page_index": meta_page,
        "metadata_snippet": meta_snip,
        "confidence": confidences
    }
    results["_meta"] = meta
    results["_evidence"] = evidence
    return results

# -----------------------
# Streamlit app
# -----------------------
st.set_page_config(page_title="FIR PII Extractor — robust", layout="wide")
st.title("FIR PII Extractor — Robust (Offline, Hindi+English)")

st.markdown(
    "Upload FIR PDF. Pipeline: PDF text-layer → EasyOCR fallback → metadata detection → per-field LLM extraction → normalization."
)

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    pdf_bytes = uploaded.read()
    with st.spinner("Extracting (text + OCR where needed)..."):
        try:
            results = extract_all_fields(pdf_bytes)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            raise

    st.success("Extraction complete — verify editable fields below.")

    # show metadata snippet and page index
    st.subheader("Metadata snippet (evidence)")
    st.text_area("Metadata snippet (auto-detected)", results["_meta"]["metadata_snippet"], height=220)
    st.write(f"Detected metadata page index: {results['_meta']['meta_page_index']}")

    st.subheader("Extracted PII (editable with evidence & confidence)")
    cols = st.columns((2, 1, 1))
    with cols[0]:
        edited = {}
        for key in SCHEMA_KEYS:
            val = results.get(key, "")
            conf = results["_meta"]["confidence"].get(key, 0.0)
            ev = results["_evidence"].get(key, "")
            st.markdown(f"**{key}** — confidence: {conf}")
            if isinstance(val, list):
                txt = "\n".join(map(str, val))
                new = st.text_area(key, value=txt, height=80)
                edited[key] = [x.strip() for x in new.splitlines() if x.strip()]
            else:
                new = st.text_input(key, value=str(val))
                edited[key] = new
            # show evidence collapsible
            if ev:
                st.expander("evidence (detected snippet)").write(ev)

    with cols[1]:
        st.subheader("Confidence summary")
        st.json(results["_meta"]["confidence"])

    with cols[2]:
        st.subheader("Quick-normalize helpers")
        if st.button("Auto-normalize phone + year"):
            # apply quick normalizations on the edited dict
            if "phone" in edited:
                edited["phone"] = normalize_phone(str(edited["phone"]))
            if "year" in edited:
                edited["year"] = normalize_year(str(edited["year"]))
            st.success("Auto-normalize applied (phone/year)")

    # Download final JSON
    if st.button("Download final JSON"):
        out = edited.copy()
        out["_meta"] = results["_meta"]
        out["_evidence"] = results["_evidence"]
        st.download_button("Download", json.dumps(out, ensure_ascii=False, indent=2), file_name="fir_pii_final.json", mime="application/json")

    st.info("If fields are wrong, correct them and press Download. Each correction can be saved to build labeled data for model improvement.")
else:
    st.info("Upload a single FIR PDF to begin. Use clean scan (>=300 DPI) for best OCR accuracy.")
