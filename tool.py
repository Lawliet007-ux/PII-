# app.py
"""
Final robust FIR PII extractor - FIXED
- PyMuPDF for text + rendering (no Poppler)
- EasyOCR for OCR fallback (en + hi)
- Deterministic extractors returning (value, confidence, evidence)
- Streamlit UI: raw text, editable fields, evidence, download JSON
"""

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import numpy as np
import easyocr
import re
import json
import dateparser
from rapidfuzz import fuzz
from typing import List, Dict, Any, Tuple

st.set_page_config(layout="wide", page_title="FIR PII Extractor — Final (Fixed)")

# ---------------------------
# Simple gazetteer (expandable)
# ---------------------------
STATE_GAZ = [
    "Maharashtra", "महाराष्ट्र", "Chhattisgarh", "छत्तीसगढ़",
    "Uttar Pradesh", "उत्तर प्रदेश", "Madhya Pradesh", "मध्य प्रदेश",
    "Delhi", "दिल्ली", "West Bengal", "पश्चिम बंगाल", "Karnataka", "कर्नाटक"
]

# label variants (English + Hindi small set)
LABELS = {
    "fir": ["fir", "f.i.r", "fir no", "fir number", "fir नं"],
    "date": ["date", "date of fir", "दिनांक", "तारीख"],
    "time": ["time", "वेळ", "समय", "Time of FIR"],
    "ps": ["police station", "p.s.", "ps", "पोलीस ठाणे", "थाना"],
    "district": ["district", "जिल्हा", "जिला"],
    "state": ["state", "राज्य"],
    "sections": ["section", "sections", "कलम"],
    "acts": ["act", "acts", "धिननयम"],
    "name": ["name", "नाव", "complainant", "informant", "accused", "आरोपी"],
    "phone": ["phone", "मोबाइल", "मोबाईल", "फोन"],
    "address": ["address", "पत्ता"]
}

# ---------------------------
# Load heavy resources once
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_easyocr_reader():
    # Creates easyocr.Reader (loads model files first run)
    return easyocr.Reader(['en', 'hi'])  # CPU by default; if you have GPU adjust accordingly

reader = load_easyocr_reader()

# ---------------------------
# Utility: render page -> PIL image
# ---------------------------
def render_page_to_image(page: fitz.Page, dpi: int = 200) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # small enhancements to help OCR
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    return img

# ---------------------------
# OCR with bboxes (EasyOCR detail=1)
# ---------------------------
def ocr_page_with_bboxes(page: fitz.Page, dpi: int = 200) -> List[Dict[str, Any]]:
    img = render_page_to_image(page, dpi=dpi)
    arr = np.array(img)
    raw = reader.readtext(arr, detail=1, paragraph=False)  # list of (bbox, text, conf) or similar
    lines = []
    for item in raw:
        # handle both tuple/list forms
        if not item:
            continue
        if len(item) == 3:
            bbox, text, conf = item
        elif len(item) == 2:
            bbox, text = item
            conf = 1.0
        else:
            # unexpected format, skip
            continue
        # bbox -> flatten to (x0,y0,x1,y1)
        try:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            bbox_flat = (min(xs), min(ys), max(xs), max(ys))
            center_y = sum(ys) / len(ys)
        except Exception:
            bbox_flat = (0, 0, 0, 0)
            center_y = 0
        lines.append({
            "text": str(text).strip(),
            "bbox": bbox_flat,
            "y": center_y,
            "conf": float(conf) if conf is not None else 1.0
        })
    # sort top -> bottom
    lines = sorted(lines, key=lambda r: r["y"])
    return lines

# ---------------------------
# Get page lines: text layer if present else OCR
# ---------------------------
def build_pages_lines_from_pdf(pdf_bytes: bytes) -> Tuple[str, List[List[Dict[str, Any]]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_lines: List[List[Dict[str, Any]]] = []
    all_texts = []
    for page in doc:
        text_layer = page.get_text("text") or ""
        text_layer = " ".join(text_layer.split())
        if text_layer and len(text_layer) > 25:
            # convert text layer -> pseudo-lines using splitlines
            lines = []
            for i, l in enumerate(text_layer.splitlines()):
                if l.strip():
                    lines.append({"text": l.strip(), "bbox": (0, i*10, 1000, i*10+8), "y": i*10, "conf": 1.0})
            pages_lines.append(lines)
            all_texts.append(text_layer)
        else:
            # fallback to OCR with bboxes
            try:
                lines = ocr_page_with_bboxes(page, dpi=250)
            except Exception as e:
                # if OCR fails, use an empty list
                lines = []
            pages_lines.append(lines)
            page_text = " ".join([ln["text"] for ln in lines])
            all_texts.append(page_text)
    full_text = "\n\n".join(all_texts)
    return full_text, pages_lines

# ---------------------------
# Small cleaning helper
# ---------------------------
def normalize_noise(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\(cid:\d+\)", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r:,.")
    return s

# ---------------------------
# Fuzzy label search in a page's lines
# ---------------------------
def fuzzy_label_search(lines: List[Dict[str, Any]], variants: List[str], cutoff: int = 70) -> Tuple[int, str, float]:
    if not lines:
        return -1, "", 0.0
    best_score = 0
    best_idx = -1
    best_text = ""
    for i, ln in enumerate(lines):
        t = ln["text"].lower()
        for v in variants:
            s = fuzz.partial_ratio(t, v.lower())
            # direct substring => perfect
            if v.lower() in t:
                s = max(s, 100)
            if s > best_score:
                best_score = s
                best_idx = i
                best_text = ln["text"]
    if best_score >= cutoff:
        return best_idx, best_text, best_score / 100.0
    return -1, "", 0.0

# ---------------------------
# Field extractors: always return (value, confidence, evidence)
# ---------------------------
def extract_fir_no(lines: List[Dict[str, Any]], full_text: str) -> Tuple[str, float, str]:
    idx, txt, score = fuzzy_label_search(lines, LABELS["fir"], cutoff=45)
    if idx >= 0:
        ln = lines[idx]["text"]
        m = re.search(r"(FIR|F\.I\.R|FIR No|FIR No\.|FIR नं)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", ln, flags=re.I)
        if m:
            return normalize_noise(m.group(2)), 0.95, ln
        # neighbors
        for j in range(idx, min(len(lines), idx + 4)):
            m2 = re.search(r"([A-Za-z0-9\-\/]{3,})", lines[j]["text"])
            if m2:
                return normalize_noise(m2.group(1)), 0.6, lines[j]["text"]
    # global search
    m = re.search(r"FIR\s*(No\.?|No|Number)?\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,})", full_text, flags=re.I)
    if m:
        return normalize_noise(m.group(2)), 0.5, m.group(0)
    return "", 0.0, ""

def extract_date_time(lines: List[Dict[str, Any]], full_text: str) -> Tuple[str, str, float, str]:
    # try to find dd/mm/yyyy or d-m-yyyy
    m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", full_text)
    if m:
        parsed = dateparser.parse(m.group(1), settings={"DATE_ORDER":"DMY"})
        if parsed:
            d = parsed.strftime("%Y-%m-%d")
            # try time
            m2 = re.search(r"(\d{1,2}:\d{2})", full_text)
            t = m2.group(1) if m2 else ""
            return d, t, 0.95, m.group(0)
    # label-based fallback
    idx, txt, score = fuzzy_label_search(lines, LABELS["date"] + LABELS["time"], cutoff=40)
    if idx >= 0:
        ln = lines[idx]["text"]
        m3 = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", ln)
        m4 = re.search(r"(\d{1,2}:\d{2})", ln)
        d = dateparser.parse(m3.group(1), settings={"DATE_ORDER":"DMY"}).strftime("%Y-%m-%d") if m3 else ""
        t = m4.group(1) if m4 else ""
        return d, t, 0.75 if d else 0.4, ln
    return "", "", 0.0, ""

def extract_phone(lines: List[Dict[str, Any]], full_text: str) -> Tuple[str, float, str]:
    m = re.search(r"(\+?\d[\d\-\s]{8,}\d)", full_text)
    if m:
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) == 10:
            return "+91" + digits, 0.95, m.group(0)
        return digits, 0.6, m.group(0)
    idx, txt, score = fuzzy_label_search(lines, LABELS["phone"], cutoff=40)
    if idx >= 0:
        m2 = re.search(r"(\+?\d[\d\-\s]{8,}\d)", lines[idx]["text"])
        if m2:
            digits = re.sub(r"\D", "", m2.group(1))
            if len(digits) == 10:
                return "+91" + digits, 0.9, lines[idx]["text"]
            return digits, 0.6, lines[idx]["text"]
    return "", 0.0, ""

def extract_sections(lines: List[Dict[str, Any]], full_text: str) -> Tuple[List[str], float, str]:
    idx, txt, score = fuzzy_label_search(lines, LABELS["sections"], cutoff=40)
    found = []
    ev = ""
    conf = 0.0
    if idx >= 0:
        ev = lines[idx]["text"]
        nums = re.findall(r"\b(\d{1,3})\b", lines[idx]["text"])
        if nums:
            found = list(dict.fromkeys(nums))
            conf = 0.9
        else:
            # neighbors
            for j in range(idx+1, min(len(lines), idx+4)):
                nums2 = re.findall(r"\b(\d{1,3})\b", lines[j]["text"])
                if nums2:
                    found.extend(nums2)
            found = list(dict.fromkeys(found))
            conf = 0.7 if found else 0.3
    else:
        # global numbers fallback
        nums = re.findall(r"\b(\d{1,3})\b", full_text)
        cand = [n for n in nums if 1 <= int(n) <= 500]
        found = list(dict.fromkeys(cand))
        conf = 0.25 if found else 0.0
        ev = "global-numeric"
    return found, conf, ev

def extract_police_station(lines: List[Dict[str, Any]], full_text: str) -> Tuple[str, float, str]:
    idx, txt, score = fuzzy_label_search(lines, LABELS["ps"], cutoff=40)
    if idx >= 0:
        ln = lines[idx]["text"]
        parts = re.split(r"[:\-]", ln, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            return normalize_noise(parts[1]), 0.9, ln
        if idx + 1 < len(lines):
            return normalize_noise(lines[idx+1]["text"]), 0.6, lines[idx+1]["text"]
        return normalize_noise(ln), 0.5, ln
    # fallback search in full text for 'P.S.' token
    m = re.search(r"\bP\.S\.\s*[:\-]?\s*([^\n,;]+)", full_text, flags=re.I)
    if m:
        return normalize_noise(m.group(1)), 0.7, m.group(0)
    return "", 0.0, ""

def extract_state_district(lines: List[Dict[str, Any]], full_text: str) -> Tuple[str, str, float, str]:
    # try "District" label first
    combined = " ".join([ln["text"] for ln in lines])
    m = re.search(r"District\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9,\- ]+)", combined, flags=re.I)
    dist = ""; state = ""; ev = ""
    if m:
        chunk = m.group(1)
        ev = m.group(0)
        parts = [p.strip() for p in re.split(r",|\(|\)|–|-", chunk) if p.strip()]
        if parts:
            dist = parts[0]
            if len(parts) > 1:
                state = parts[-1]
    # fuzzy-match state from full_text
    if not state:
        best_s = None; best_score = 0
        for s in STATE_GAZ:
            sc = fuzz.WRatio(full_text.lower(), s.lower())
            if sc > best_score:
                best_score = sc; best_s = s
        if best_score > 60:
            state = best_s; ev = "global-state-match"
    conf = 0.6 if (dist or state) else 0.0
    return normalize_noise(dist), normalize_noise(state), conf, ev

def extract_names(lines: List[Dict[str, Any]], full_text: str) -> Tuple[List[str], float, str]:
    names = []
    ev = ""
    idx, txt, score = fuzzy_label_search(lines, LABELS["name"], cutoff=35)
    if idx >= 0:
        ln = lines[idx]["text"]
        parts = re.split(r"[:\-]\s*", ln, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            names.append(parts[1].strip()); ev = ln
    # also scan for honorifics or capitalized words (simple heuristic)
    for ln in lines[:30]:
        t = ln["text"].strip()
        if re.search(r"\b(Shri|Shri\.|Smt|Mr\.|Mrs\.)\b", t, flags=re.I):
            # extract name-like part after honorific
            m = re.search(r"(Shri\.?|Shri|Smt|Mr\.|Mrs\.)\s*([A-Za-z\u0900-\u097F\s\.]{3,})", t)
            if m:
                cand = m.group(2).strip()
                if cand and cand not in names:
                    names.append(cand)
    if names:
        return [normalize_noise(n) for n in names], 0.6, ev or "heuristic"
    return [], 0.0, ""

def extract_address(lines: List[Dict[str, Any]], full_text: str) -> Tuple[str, float, str]:
    idx, txt, score = fuzzy_label_search(lines, LABELS["address"], cutoff=40)
    if idx >= 0:
        # take this line + next two for multi-line addresses
        addr = lines[idx]["text"]
        for j in range(idx+1, min(idx+3, len(lines))):
            addr += " " + lines[j]["text"]
        return normalize_noise(addr), 0.6, lines[idx]["text"]
    # fallback: look for "Address" in full_text
    m = re.search(r"(Address|पत्ता)\s*[:\-]?\s*([^\n]{10,200})", full_text, flags=re.I)
    if m:
        return normalize_noise(m.group(2)), 0.5, m.group(0)
    return "", 0.0, ""

# ---------------------------
# Main orchestrator
# ---------------------------
def extract_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    try:
        full_text, pages_lines = build_pages_lines_from_pdf(pdf_bytes)
    except Exception as exc:
        raise RuntimeError(f"Unable to process PDF: {exc}")

    # choose best page for metadata: highest label matches
    best_idx = 0; best_score = -1
    label_variants = [v for vals in LABELS.values() for v in vals]
    for i, pl in enumerate(pages_lines):
        tally = 0
        lower = " ".join([ln["text"].lower() for ln in pl])
        for v in label_variants:
            if v.lower() in lower:
                tally += 1
        if tally > best_score:
            best_score = tally; best_idx = i
    page_lines = pages_lines[best_idx] if pages_lines else []

    # extract fields
    fir_val, fir_conf, fir_ev = extract_fir_no(page_lines, full_text)
    date_val, time_val, dt_conf, dt_ev = extract_date_time(page_lines, full_text)
    phone_val, phone_conf, phone_ev = extract_phone(page_lines, full_text)
    sections_val, sec_conf, sec_ev = extract_sections(page_lines, full_text)
    ps_val, ps_conf, ps_ev = extract_police_station(page_lines, full_text)
    dist_val, state_val, sd_conf, sd_ev = extract_state_district(page_lines, full_text)
    names_val, names_conf, names_ev = extract_names(page_lines, full_text)
    addr_val, addr_conf, addr_ev = extract_address(page_lines, full_text)

    result = {
        "fir_no": fir_val,
        "date": date_val,
        "time": time_val,
        "phone": phone_val,
        "police_station": ps_val,
        "dist_name": dist_val,
        "state_name": state_val,
        "under_sections": sections_val,
        "under_acts": [],  # acts extraction left empty (could be added like sections)
        "names": names_val,
        "address": addr_val,
        "_meta": {"best_metadata_page": best_idx},
        "_evidence": {
            "fir_no": fir_ev, "date": dt_ev, "time": dt_ev, "phone": phone_ev,
            "police_station": ps_ev, "state_district": sd_ev, "sections": sec_ev,
            "names": names_ev, "address": addr_ev
        },
        "_confidence": {
            "fir_no": round(float(fir_conf), 2),
            "date": round(float(dt_conf), 2),
            "time": round(float(0.8 if time_val else 0.0), 2),
            "phone": round(float(phone_conf), 2),
            "police_station": round(float(ps_conf), 2),
            "state_district": round(float(sd_conf), 2),
            "sections": round(float(sec_conf), 2),
            "names": round(float(names_conf), 2),
            "address": round(float(addr_conf), 2)
        }
    }
    return result

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("FIR PII Extractor — Final (Fixed)")

st.markdown(
    "Upload a single FIR PDF (scan or digital). The tool uses PDF text-layer when available and EasyOCR fallback. "
    "It returns per-field values with confidence and evidence. Edit fields and download final JSON."
)

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    with st.spinner("Running OCR + extraction (this can take a few seconds)..."):
        try:
            extracted = extract_from_pdf_bytes(pdf_bytes)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            raise

    st.success("Extraction finished — review outputs below.")

    st.subheader("Detected metadata page (index)")
    st.write(extracted.get("_meta", {}))

    st.subheader("Extracted fields (editable)")
    cols = st.columns([3,1,1])
    edited = {}
    with cols[0]:
        edited["fir_no"] = st.text_input("FIR No", value=extracted.get("fir_no",""))
        edited["date"] = st.text_input("Date (YYYY-MM-DD)", value=extracted.get("date",""))
        edited["time"] = st.text_input("Time", value=extracted.get("time",""))
        edited["police_station"] = st.text_input("Police Station", value=extracted.get("police_station",""))
        edited["state_name"] = st.text_input("State", value=extracted.get("state_name",""))
        edited["dist_name"] = st.text_input("District", value=extracted.get("dist_name",""))
        edited["phone"] = st.text_input("Phone", value=extracted.get("phone",""))
        edited["address"] = st.text_area("Address", value=extracted.get("address",""), height=100)
        edited["under_sections"] = st.text_area("Under Sections (one per line)", value="\n".join(extracted.get("under_sections",[])), height=120)
        edited["names"] = st.text_area("Names (one per line)", value="\n".join(extracted.get("names",[])), height=120)

    with cols[1]:
        st.subheader("Confidence")
        st.json(extracted.get("_confidence", {}))
        st.subheader("Evidence")
        st.json(extracted.get("_evidence", {}))

    with cols[2]:
        st.subheader("Actions")
        if st.button("Auto-normalize phone/date/sections"):
            # phone normalization
            p = edited.get("phone","")
            digits = re.sub(r"[^\d]", "", p)
            if len(digits) == 10:
                edited["phone"] = "+91" + digits
            elif len(digits) == 11 and digits.startswith("0"):
                edited["phone"] = "+91" + digits[1:]
            # date normalization
            if edited.get("date"):
                d = dateparser.parse(edited["date"], settings={"DATE_ORDER":"DMY"})
                if d:
                    edited["date"] = d.strftime("%Y-%m-%d")
            # sections normalization
            secs = [re.sub(r"[^\d]", "", s) for s in edited.get("under_sections","").splitlines()]
            secs = [s for s in secs if s]
            edited["under_sections"] = "\n".join(secs)
            st.success("Auto-normalize applied.")

        if st.button("Download final JSON"):
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
                "names": [s for s in edited.get("names","").splitlines() if s.strip()],
                "_meta": extracted.get("_meta", {}),
                "_evidence": extracted.get("_evidence", {}),
                "_confidence": extracted.get("_confidence", {})
            }
            st.download_button("Download final JSON", json.dumps(out, ensure_ascii=False, indent=2),
                               file_name="fir_pii_final.json", mime="application/json")
else:
    st.info("Upload one FIR PDF to run extraction. Use good-quality scans (>=300 DPI) for best OCR accuracy.")
