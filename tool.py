# app.py
"""
Robust FIR PII Extractor (final)
- Works offline (no external API keys)
- Python 3.13 safe: uses EasyOCR (no cv2)
- Uses OCR bboxes, fuzzy label detection, specialized extractors, and optional LLM normalization
- Shows evidence + per-field confidence, editable UI + JSON download
"""

import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import easyocr
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from rapidfuzz import process, fuzz
import re
import json
import dateparser
from typing import List, Tuple, Dict, Any
import html
import json5

# ---------------------------
# Config & small gazetteers
# ---------------------------
st.set_page_config(layout="wide", page_title="FIR PII Extractor (Robust Final)")

# Minimal state list + common Hindi forms — expand offline if needed
ALL_STATES = [
    "Maharashtra", "महाराष्ट्र", "Chhattisgarh", "छत्तीसगढ़", "Uttar Pradesh", "उत्तर प्रदेश",
    "Madhya Pradesh", "मध्य प्रदेश", "Karnataka", "कर्नाटक", "Tamil Nadu", "तमिलनाडु",
    "Delhi", "दिल्ली", "West Bengal", "पश्चिम बंगाल"
]

LABEL_KEYWORDS = {
    "fir_no": ["fir no", "fir", "fir no.", "fir नं", "fir number", "f.i.r", "f.i.r."],
    "date": ["date", "date of fir", "date :", "दिनांक", "दिनांक :", "दिनांक आणि वेळ"],
    "time": ["time", "time :", "वेळ", "वेळ :"],
    "police_station": ["police station", "p.s.", "p.s", "पोलीस ठाणे", "पोलीस ठाणे:"],
    "district": ["district", "district (state)", "ज जिल्हा", "जिल्हा", "जिल्हा:"],
    "state": ["state", "राज्य", "राज्य:"],
    "sections": ["sections", "sections (कलम)", "कलम", "sections:"],
    "acts": ["acts", "act", "धिननयम", "act(s)"],
    "name": ["name", "नाव", "complainant", "informant", "accused", "नाव :"],
    "phone": ["phone", "मोबाईल", "मोबाइल", "फोन", "phone no", "मो.नं"],
    "address": ["address", "पत्ता", "address:"],
}

# ---------------------------
# Helpers: image preprocessing (Pillow)
# ---------------------------
def enhance_image_for_ocr(pil_img: Image.Image) -> Image.Image:
    # convert to RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    # increase contrast, sharpen a bit
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    # convert to grayscale then back if needed by EasyOCR (it accepts RGB arrays)
    return pil_img

# ---------------------------
# Load OCR (EasyOCR) and optional LLM normalizer
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    reader = easyocr.Reader(['en', 'hi'])  # loads models once
    # LLM normalization is optional and can be commented if you don't want to use it
    # We leave it out by default for maximum determinism; you can add a transformers pipeline if desired.
    return reader

reader = load_resources()

# ---------------------------
# OCR page -> lines with bbox
# ---------------------------
def ocr_pdf_pages(pdf_bytes: bytes, dpi: int = 300) -> List[List[Dict[str, Any]]]:
    """
    Returns list-of-pages where each page is list of dicts: {"text":..., "bbox":..., "y":...}
    Uses EasyOCR for OCR, and PyMuPDF for rendering instead of pdf2image (no Poppler needed).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_results = []
    for page in doc:
        # Render page to image
        zoom = dpi / 72  # scale factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = enhance_image_for_ocr(img)
        arr = np.array(img)
        raw = reader.readtext(arr, detail=1, paragraph=False)  # list of [bbox, text, conf]

        lines = []
        for bbox, text, conf in raw:
            ys = [p[1] for p in bbox]
            x_vals = [p[0] for p in bbox]
            bbox_flat = (min(x_vals), min(ys), max(x_vals), max(ys))
            center_y = sum(ys) / len(ys)
            lines.append({"text": text.strip(), "bbox": bbox_flat, "y": center_y, "conf": conf})
        lines = sorted(lines, key=lambda r: r["y"])
        pages_results.append(lines)
    return pages_results


# ---------------------------
# Helper: join lines into paragraphs by vertical distance
# ---------------------------
def lines_to_paragraphs(lines: List[Dict[str, Any]], y_gap_thresh: float = 15.0) -> List[str]:
    if not lines:
        return []
    paragraphs = []
    cur = [lines[0]["text"]]
    prev_y = lines[0]["y"]
    for ln in lines[1:]:
        if (ln["y"] - prev_y) > y_gap_thresh:
            paragraphs.append(" ".join(cur))
            cur = [ln["text"]]
        else:
            cur.append(ln["text"])
        prev_y = ln["y"]
    if cur:
        paragraphs.append(" ".join(cur))
    return paragraphs

# ---------------------------
# Fuzzy label search
# ---------------------------
def fuzzy_find_label(lines: List[Dict[str, Any]], label_variants: List[str], score_cutoff: int = 70) -> Tuple[int, str, float]:
    """
    Returns (line_index, matched_text, score) for the best match among lines for any of label_variants.
    If none found returns (-1, "", 0.0).
    """
    texts = [l["text"] for l in lines]
    best_score = 0
    best_idx = -1
    best_match = ""
    for idx, t in enumerate(texts):
        lower = t.lower()
        # test each variant
        for v in label_variants:
            score = fuzz.partial_ratio(lower, v.lower())
            # also test reversed: v in t
            if v.lower() in lower:
                score = max(score, 100)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_match = t
    if best_score >= score_cutoff:
        return best_idx, best_match, best_score / 100.0
    return -1, "", 0.0

# ---------------------------
# Extractors for fields
# ---------------------------
def extract_fir_no_from_lines(lines: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    # search near label
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["fir_no"], score_cutoff=45)
    if idx >= 0:
        candidate = text
        # try to extract numbers from same line (common patterns)
        m = re.search(r"(fir|f\.i\.r|fir no|fir no\.?)\W*[:\-]?\s*([\w\-/]+)", text, flags=re.I)
        if m and m.group(2):
            return m.group(2).strip(), 0.9, text
        # try digits inside text
        m2 = re.search(r"([A-Za-z0-9\-\/]{2,})", text)
        if m2:
            return m2.group(1).strip(), 0.6, text
        # else search neighboring lines (next 3 lines)
        for j in range(idx, min(idx+4, len(lines))):
            m3 = re.search(r"([A-Za-z0-9\-/]{2,})", lines[j]["text"])
            if m3:
                return m3.group(1).strip(), 0.5, lines[j]["text"]
    # fallback: global search for patterns anywhere
    joined = " ".join([ln["text"] for ln in lines])
    m = re.search(r"FIR\s*(No\.?|No|#)?\s*[:\-]?\s*([A-Za-z0-9\-\/]{2,})", joined, flags=re.I)
    if m:
        return m.group(2).strip(), 0.5, m.group(0)
    return "", 0.0, ""

def extract_date_from_lines(lines: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    # search for date patterns dd/mm/yyyy or variants
    joined = " ".join([ln["text"] for ln in lines])
    # try dd/mm/yyyy first
    m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", joined)
    if m:
        d = dateparser.parse(m.group(1), settings={"DATE_ORDER": "DMY"})
        if d:
            return d.strftime("%Y-%m-%d"), 0.95, m.group(1)
    # search for 'Date' label
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["date"], score_cutoff=40)
    if idx >= 0:
        # try extract date from this line
        m2 = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", lines[idx]["text"])
        if m2:
            d = dateparser.parse(m2.group(1), settings={"DATE_ORDER": "DMY"})
            if d:
                return d.strftime("%Y-%m-%d"), 0.9, lines[idx]["text"]
        # else look nearby lines
        for j in range(max(0, idx-2), min(len(lines), idx+3)):
            m3 = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", lines[j]["text"])
            if m3:
                d = dateparser.parse(m3.group(1), settings={"DATE_ORDER": "DMY"})
                if d:
                    return d.strftime("%Y-%m-%d"), 0.85, lines[j]["text"]
    return "", 0.0, ""

def extract_time_from_lines(lines: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    joined = " ".join([ln["text"] for ln in lines])
    # match HH:MM optionally with AM/PM or 24h
    m = re.search(r"(\d{1,2}[:]\d{2})(?:\s*(AM|PM|am|pm))?", joined)
    if m:
        return m.group(1), 0.9, m.group(0)
    # fallback: search label 'time'
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["time"], score_cutoff=40)
    if idx >= 0:
        m2 = re.search(r"(\d{1,2}[:]\d{2})", lines[idx]["text"])
        if m2:
            return m2.group(1), 0.8, lines[idx]["text"]
    return "", 0.0, ""

def extract_phone_from_lines(lines: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    joined = " ".join([ln["text"] for ln in lines])
    # find 10+ digit sequences
    m = re.search(r"(\+?\d[\d\-\s]{8,}\d)", joined)
    if m:
        phone = re.sub(r"[^\d+]", "", m.group(1))
        # normalize
        digits = re.sub(r"[^\d]", "", phone)
        if len(digits) == 10:
            phone = "+91" + digits
        elif len(digits) == 11 and digits.startswith("0"):
            phone = "+91" + digits[1:]
        else:
            phone = "+" + digits if not phone.startswith("+") else phone
        return phone, 0.95, m.group(0)
    # label-based fallback
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["phone"], score_cutoff=40)
    if idx >= 0:
        m2 = re.search(r"(\+?\d[\d\-\s]{8,}\d)", lines[idx]["text"])
        if m2:
            digits = re.sub(r"[^\d]", "", m2.group(1))
            if len(digits) == 10:
                return "+91" + digits, 0.9, lines[idx]["text"]
            return m2.group(1), 0.6, lines[idx]["text"]
    return "", 0.0, ""

def extract_sections_from_lines(lines: List[Dict[str, Any]]) -> Tuple[List[str], float, str]:
    # find 'sections' label
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["sections"], score_cutoff=40)
    found = []
    evidence = ""
    conf = 0.0
    if idx >= 0:
        evidence = lines[idx]["text"]
        # extract numbers from same line
        nums = re.findall(r"\d{1,4}", lines[idx]["text"])
        if nums:
            found = list(dict.fromkeys(nums))  # dedupe preserving order
            conf = 0.9
        else:
            # check next lines
            for j in range(idx+1, min(idx+4, len(lines))):
                nums2 = re.findall(r"\d{1,4}", lines[j]["text"])
                if nums2:
                    found.extend(nums2)
                    evidence += " | " + lines[j]["text"]
            found = list(dict.fromkeys(found))
            conf = 0.7 if found else 0.3
    else:
        # global numeric heuristics: collect plausible section numbers that often accompany "sections" words
        joined = " ".join([ln["text"] for ln in lines])
        nums = re.findall(r"\b(\d{1,4})\b", joined)
        # filter likely section numbers (remove year-like 2025 etc)
        cand = [n for n in nums if 1 <= len(n) <= 3 and int(n) < 1000]
        cand = list(dict.fromkeys(cand))
        if cand:
            found = cand[:15]
            conf = 0.25
            evidence = "global-numeric"
    return found, conf, evidence

def extract_police_station(lines: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["police_station"], score_cutoff=40)
    if idx >= 0:
        # try to extract text after colon or after label
        ln = lines[idx]["text"]
        m = re.split(r"[:\-]\s*", ln, maxsplit=1)
        if len(m) > 1 and len(m[1].strip()) > 1:
            return m[1].strip(), 0.9, ln
        # else try neighbor lines
        if idx+1 < len(lines):
            return lines[idx+1]["text"].strip(), 0.6, lines[idx+1]["text"]
        return ln.strip(), 0.5, ln
    # fallback: search nearby 'P.S.' tokens in text
    for i, ln in enumerate(lines):
        if re.search(r"\bP\.S\.|\bPS\b|\bपोलीस ठाणे\b", ln["text"], flags=re.I):
            # get few tokens following the label
            parts = re.split(r"P\.S\.|PS|पोलीस ठाणे", ln["text"], flags=re.I)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip(" :,-"), 0.6, ln["text"]
    return "", 0.0, ""

def extract_state_and_district(lines: List[Dict[str, Any]]) -> Tuple[str, str, float, str]:
    # search for district label
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["district"], score_cutoff=40)
    state_val, dist_val, conf, ev = "", "", 0.0, ""
    if idx >= 0:
        ev = lines[idx]["text"]
        # try to parse "District (State): <dist> , <state>"
        m = re.split(r":", lines[idx]["text"], maxsplit=1)
        if len(m) > 1:
            after = m[1]
            parts = [p.strip() for p in re.split(r",|,|\|", after) if p.strip()]
            if parts:
                dist_val = parts[0]
                if len(parts) > 1:
                    state_val = parts[-1]
            else:
                dist_val = after.strip()
        else:
            # neighbor line heuristic
            if idx+1 < len(lines):
                candidate = lines[idx+1]["text"]
                # split by comma
                parts = [p.strip() for p in re.split(r",|,|\|", candidate) if p.strip()]
                if parts:
                    dist_val = parts[0]
                    if len(parts) > 1:
                        state_val = parts[-1]
        conf = 0.6 if dist_val or state_val else 0.2
        # fuzzy-match state to gazetteer
        if state_val:
            state_match, score = process.extractOne(state_val, ALL_STATES, scorer=fuzz.WRatio) or ("", 0)
            if score > 65:
                state_val = state_match
                conf = max(conf, score/100.0)
    else:
        # global fallback: fuzzy match every line to states gazetteer
        texts = [ln["text"] for ln in lines]
        best = process.extractOne(" ".join(texts), ALL_STATES, scorer=fuzz.WRatio)
        if best and best[1] > 60:
            state_val = best[0]
            conf = best[1]/100.0
            ev = "global search"
    return state_val, dist_val, conf, ev

def extract_names(lines: List[Dict[str, Any]]) -> Tuple[List[str], float, str]:
    # Look for 'Complainant / Informant' or 'Name' labels
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["name"], score_cutoff=35)
    names = []
    ev = ""
    conf = 0.0
    if idx >= 0:
        ev = lines[idx]["text"]
        # try to extract names after "Name" tokens
        m = re.split(r"Name|नाव|Complainant|Informant|आरोपी|Accused", lines[idx]["text"], flags=re.I)
        if len(m) > 1 and m[1].strip():
            candidate = m[1].strip(" :,-.")
            names.append(candidate)
            conf = 0.7
        else:
            # scan next few lines for capitalized words / Devanagari sequences
            for j in range(idx, min(idx+6, len(lines))):
                candidate = lines[j]["text"].strip()
                if len(candidate) > 2 and any(ord(ch) > 128 for ch in candidate) or re.search(r"[A-Z]{2,}", candidate):
                    names.append(candidate)
            names = list(dict.fromkeys(names))
            conf = 0.5 if names else 0.0
    else:
        # global heuristic: look for typical name-like lines with alphabets and capitals
        for ln in lines:
            if re.search(r"\b(Name|नाव)\b", ln["text"], flags=re.I):
                # same as above
                parts = re.split(r"Name|नाव", ln["text"], flags=re.I)
                if len(parts) > 1 and parts[1].strip():
                    names.append(parts[1].strip())
        names = list(dict.fromkeys(names))
        conf = 0.4 if names else 0.0
    return names, conf, ev

def extract_address(lines: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    idx, text, score = fuzzy_find_label(lines, LABEL_KEYWORDS["address"], score_cutoff=40)
    if idx >= 0:
        # try to grab line and following lines until blank or break
        addr = lines[idx]["text"]
        for j in range(idx+1, min(idx+4, len(lines))):
            if len(lines[j]["text"]) > 3:
                addr += " " + lines[j]["text"]
        return addr.strip(), 0.7, addr
    # fallback: find lines with typical address tokens or city names
    joined = " ".join([ln["text"] for ln in lines])
    m = re.search(r"Address[:\s\-]*(.*?)(?:District|District:|District \(|$)", joined, flags=re.I)
    if m:
        return m.group(1).strip(), 0.5, m.group(0)
    return "", 0.0, ""

# ---------------------------
# Orchestrate extraction for a PDF (prefer first page metadata)
# ---------------------------
def extract_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    pages = ocr_pdf_pages(pdf_bytes, dpi=300)
    # pick page 0 (usually) but detect header via LABEL_KEYWORDS
    # find page with highest count of header keywords
    page_scores = []
    for pidx, plines in enumerate(pages):
        txt = " ".join([ln["text"] for ln in plines]).lower()
        score = sum(1 for kw in sum(LABEL_KEYWORDS.values(), []) if kw in txt)
        page_scores.append((score, pidx))
    page_scores.sort(reverse=True)
    # choose best page if non-zero else first page
    best_page_idx = page_scores[0][1] if page_scores and page_scores[0][0] > 0 else 0
    lines = pages[best_page_idx] if pages else []
    paragraphs = lines_to_paragraphs(lines, y_gap_thresh=12.0)

    # Keep also whole-document lines for fallback
    all_lines = []
    for pl in pages:
        all_lines.extend(pl)

    # extract fields
    out = {}
    evidences = {}
    confidences = {}

    fir_no, c_fir, ev_fir = extract_fir_no_from_lines(lines)
    out["fir_no"], evidences["fir_no"], confidences["fir_no"] = fir_no, ev_fir, c_fir

    date_val, c_date, ev_date = extract_date_from_lines(lines)
    out["date"], evidences["date"], confidences["date"] = date_val, ev_date, c_date

    time_val, c_time, ev_time = extract_time_from_lines(lines)
    out["time"], evidences["time"], confidences["time"] = time_val, ev_time, c_time

    phone, c_phone, ev_phone = extract_phone_from_lines(all_lines)
    out["phone"], evidences["phone"], confidences["phone"] = phone, ev_phone, c_phone

    ps, c_ps, ev_ps = extract_police_station(lines)
    out["police_station"], evidences["police_station"], confidences["police_station"] = ps, ev_ps, c_ps

    state, dist, c_sd, ev_sd = extract_state_and_district(lines)
    out["state_name"], out["dist_name"], evidences["state_dist"], confidences["state_dist"] = state, dist, ev_sd, c_sd

    sections, c_sec, ev_sec = extract_sections_from_lines(lines)
    out["under_sections"], evidences["under_sections"], confidences["under_sections"] = sections, ev_sec, c_sec

    acts, c_act, ev_act = [], 0.0, ""
    # try to find acts label
    idx_act, act_text, s_act = fuzzy_find_label(lines, LABEL_KEYWORDS["acts"], score_cutoff=40)
    if idx_act >= 0:
        acts_candidates = re.split(r"[,\|;/]+", lines[idx_act]["text"])
        acts = [a.strip() for a in acts_candidates if len(a.strip())>1]
        ev_act = lines[idx_act]["text"]
        c_act = 0.6
    out["under_acts"], evidences["under_acts"], confidences["under_acts"] = acts, ev_act, c_act

    names, c_names, ev_names = extract_names(lines)
    out["names"], evidences["names"], confidences["names"] = names, ev_names, c_names

    address, c_addr, ev_addr = extract_address(lines)
    out["address"], evidences["address"], confidences["address"] = address, ev_addr, c_addr

    # Add meta info and full-text for evidence
    out["_meta"] = {
        "best_page_index": best_page_idx,
        "paragraphs": paragraphs,
        "full_pages_count": len(pages),
    }
    out["_evidence"] = evidences
    out["_confidence"] = confidences

    return out

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("FIR PII Extractor — Robust Final (Offline)")

st.markdown(
    "Drop a scanned FIR PDF (Hindi/English). The tool uses EasyOCR + fuzzy label matching + dedicated extractors "
    "for dates, phone, sections, names, police station and address. Edit fields, verify evidence, download JSON."
)

uploaded = st.file_uploader("Upload FIR (PDF)", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    with st.spinner("Running OCR and extraction (this can take a few seconds)..."):
        try:
            results = extract_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            st.error("Extraction failed: " + str(e))
            raise

    st.success("Extraction done — review results below.")

    # Show metadata snippet: top paragraphs
    st.subheader("Detected metadata paragraphs (evidence)")
    for i, p in enumerate(results["_meta"]["paragraphs"][:6]):
        st.markdown(f"**Paragraph {i+1}**: {p}")

    st.subheader("Extracted fields (editable)")
    cols = st.columns([3, 1, 1])
    edited = {}
    with cols[0]:
        # Editable fields
        edited["fir_no"] = st.text_input("FIR No", value=str(results.get("fir_no","")))
        edited["date"] = st.text_input("Date (YYYY-MM-DD)", value=str(results.get("date","")))
        edited["time"] = st.text_input("Time (HH:MM)", value=str(results.get("time","")))
        edited["police_station"] = st.text_input("Police Station", value=str(results.get("police_station","")))
        edited["state_name"] = st.text_input("State", value=str(results.get("state_name","")))
        edited["dist_name"] = st.text_input("District", value=str(results.get("dist_name","")))
        edited["phone"] = st.text_input("Phone", value=str(results.get("phone","")))
        edited["address"] = st.text_area("Address", value=str(results.get("address","")), height=100)
        edited["under_sections"] = st.text_area("Under Sections (one per line)", value="\n".join(results.get("under_sections",[])), height=100)
        edited["under_acts"] = st.text_area("Under Acts (one per line)", value="\n".join(results.get("under_acts",[])), height=80)
        edited["names"] = st.text_area("Names (one per line)", value="\n".join(results.get("names",[])), height=100)

    with cols[1]:
        st.subheader("Confidence")
        st.json(results.get("_confidence", {}))
        st.markdown("**Evidence mapping**")
        st.json(results.get("_evidence", {}))

    with cols[2]:
        st.subheader("Actions")
        if st.button("Auto-normalize phone/year/sections"):
            # normalize phone and date format
            phone_raw = edited.get("phone","")
            if phone_raw:
                digits = re.sub(r"[^\d]", "", phone_raw)
                if len(digits) == 10:
                    edited["phone"] = "+91" + digits
                elif len(digits) == 11 and digits.startswith("0"):
                    edited["phone"] = "+91" + digits[1:]
                else:
                    edited["phone"] = phone_raw
            # normalize date input: try parse
            if edited.get("date"):
                d = dateparser.parse(edited["date"], settings={"DATE_ORDER":"DMY"})
                if d:
                    edited["date"] = d.strftime("%Y-%m-%d")
            # normalize sections: split lines, digits only
            secs = [re.sub(r"[^\d]", "", s) for s in edited.get("under_sections","").splitlines()]
            secs = [s for s in secs if s]
            edited["under_sections"] = "\n".join(secs)
            st.success("Auto-normalize applied.")

        if st.button("Download JSON"):
            out = {
                "fir_no": edited.get("fir_no",""),
                "date": edited.get("date",""),
                "time": edited.get("time",""),
                "police_station": edited.get("police_station",""),
                "state_name": edited.get("state_name",""),
                "dist_name": edited.get("dist_name",""),
                "phone": edited.get("phone",""),
                "address": edited.get("address",""),
                "under_sections": [s for s in edited.get("under_sections","").splitlines() if s.strip()],
                "under_acts": [s for s in edited.get("under_acts","").splitlines() if s.strip()],
                "names": [s for s in edited.get("names","").splitlines() if s.strip()],
                "_meta": results.get("_meta", {}),
                "_evidence": results.get("_evidence", {}),
                "_confidence": results.get("_confidence", {})
            }
            st.download_button("Download final JSON", data=json.dumps(out, ensure_ascii=False, indent=2),
                               file_name="fir_pii_final.json", mime="application/json")

    st.info("If fields are inaccurate, correct them above and download final JSON. Corrections are valuable training data.")

else:
    st.info("Upload one FIR PDF to extract PII. Use good-quality scans at >= 300 DPI for best results.")
