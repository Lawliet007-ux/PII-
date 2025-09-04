# app.py
"""
Final robust FIR PII extractor.
- Works on Python 3.13 (no cv2 / no poppler)
- Uses PyMuPDF for rendering + text-layer; EasyOCR for OCR fallback with bboxes
- Deterministic extractors + optional HF token-classification NER fusion
- Evidence + confidence + editable UI + JSON download
"""

import streamlit as st
import fitz  # PyMuPDF
import easyocr
from PIL import Image, ImageEnhance
import numpy as np
import re
import json
import dateparser
from typing import List, Dict, Any, Tuple, Optional
from rapidfuzz import process, fuzz

# Optional HF token-classification (used if transformers available & model can be downloaded)
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -----------------------
# Config
# -----------------------
st.set_page_config(layout="wide", page_title="FIR PII Extractor — Final")
STATE_GAZ = [
    "Maharashtra","महाराष्ट्र","Chhattisgarh","छत्तीसगढ़","Uttar Pradesh","उत्तर प्रदेश",
    "Madhya Pradesh","मध्य प्रदेश","Delhi","दिल्ली","West Bengal","पश्चिम बंगाल"
]

LABEL_MAP = {
    "fir": ["fir no", "fir", "fir no.", "f.i.r", "fir number", "fir नं"],
    "date": ["date", "date of fir", "दिनांक", "तारीख"],
    "time": ["time", "वेळ", "समय"],
    "ps": ["police station", "p.s.", "पोलीस ठाणे", "थाना"],
    "district": ["district", "जिल्हा", "जिला"],
    "state": ["state", "राज्य"],
    "sections": ["section", "sections", "कलम"],
    "acts": ["act", "acts", "धिननयम"],
    "name": ["name", "नाव", "complainant", "informant", "accused", "आरोपी"],
    "phone": ["phone", "मोबाइल", "फोन", "मो.नं", "मोबाईल"],
    "address": ["address", "पत्ता"]
}

# -----------------------
# Load OCR & optional NER
# -----------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    reader = easyocr.Reader(['en', 'hi'])
    ner = None
    if HF_AVAILABLE:
        try:
            # token-classification model - uses xlm-roberta or similar
            ner = hf_pipeline("token-classification", grouped_entities=True, device=-1)
        except Exception:
            ner = None
    return reader, ner

reader, hf_ner = load_resources()

# -----------------------
# Utilities
# -----------------------
def render_page_to_pil(page, dpi=200) -> Image.Image:
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.05)
    return img

def ocr_page_with_bboxes(page, dpi=200) -> List[Dict[str, Any]]:
    """
    Returns list of line dicts: {"text":..., "bbox":(x0,y0,x1,y1), "y":center_y, "conf":...}
    """
    img = render_page_to_pil(page, dpi=dpi)
    arr = np.array(img)
    raw = reader.readtext(arr, detail=1, paragraph=False)  # [bbox, text, conf]
    lines = []
    for bbox, text, conf in raw:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        bbox_flat = (min(xs), min(ys), max(xs), max(ys))
        center_y = sum(ys)/len(ys)
        lines.append({"text": text.strip(), "bbox": bbox_flat, "y": center_y, "conf": conf})
    # sort top->bottom
    lines = sorted(lines, key=lambda r: r["y"])
    return lines

def get_text_layer_or_ocr(doc) -> Tuple[str, List[List[Dict[str, Any]]]]:
    """
    Returns (full_text, pages_lines). pages_lines = list per page of OCR lines (dicts).
    Uses text layer if present; if page text is empty, uses OCR with bboxes.
    """
    pages_lines = []
    all_text = []
    for page in doc:
        txt = page.get_text("text") or ""
        txt = " ".join(txt.split())
        if txt and len(txt) > 20:
            # build simple lines from text layer (splitlines)
            lines = []
            for i, l in enumerate(txt.splitlines()):
                if l.strip():
                    lines.append({"text": l.strip(), "bbox": (0, i*10, 1000, i*10+8), "y": i*10, "conf": 1.0})
            pages_lines.append(lines)
            all_text.append(txt)
        else:
            # OCR fallback with bboxes
            lines = ocr_page_with_bboxes(page, dpi=250)
            pages_lines.append(lines)
            page_text = " ".join([l["text"] for l in lines])
            all_text.append(page_text)
    return "\n\n".join(all_text), pages_lines

def paragraphs_from_lines(lines: List[Dict[str, Any]], y_gap=15.0) -> List[str]:
    if not lines: return []
    paras = []
    cur = [lines[0]["text"]]
    prev_y = lines[0]["y"]
    for ln in lines[1:]:
        if ln["y"] - prev_y > y_gap:
            paras.append(" ".join(cur))
            cur = [ln["text"]]
        else:
            cur.append(ln["text"])
        prev_y = ln["y"]
    if cur: paras.append(" ".join(cur))
    return paras

def fuzzy_label_search(page_lines: List[Dict[str, Any]], variants: List[str], score_cutoff=65) -> Tuple[int,str,float]:
    texts = [ln["text"] for ln in page_lines]
    best_score = 0; best_idx = -1; best_txt = ""
    for i, t in enumerate(texts):
        low = t.lower()
        for v in variants:
            score = fuzz.partial_ratio(low, v.lower())
            if v.lower() in low:
                score = max(score, 100)
            if score > best_score:
                best_score = score; best_idx = i; best_txt = t
    if best_score >= score_cutoff:
        return best_idx, best_txt, best_score/100.0
    return -1, "", 0.0

# -----------------------
# Field extractors (deterministic + layout)
# -----------------------
def extract_fir_no_from_page(lines: List[Dict[str, Any]]) -> Tuple[str,float,str]:
    idx, txt, sc = fuzzy_label_search(lines, LABEL_MAP["fir"], score_cutoff=45)
    if idx >= 0:
        # attempt to take after ":" or nearby tokens
        m = re.search(r"(FIR|F\.I\.R|फिर)[^\dA-Za-z\-]*(?P<num>[A-Za-z0-9\-\/]+)", txt, flags=re.I)
        if m:
            return m.group("num"), 0.95, txt
        # else search neighbors
        for j in range(idx, min(idx+4, len(lines))):
            m2 = re.search(r"([A-Za-z0-9\-\/]{3,})", lines[j]["text"])
            if m2:
                return m2.group(1), 0.6, lines[j]["text"]
    # global regex
    whole = " ".join([l["text"] for l in lines])
    m = re.search(r"\b([A-Za-z0-9]{4,}[-/][A-Za-z0-9\-\/]*)\b", whole)
    if m:
        return m.group(1), 0.45, m.group(0)
    return "", 0.0, ""

def extract_date_time(lines: List[Dict[str, Any]]) -> Tuple[str,str,float,str]:
    whole = " ".join([l["text"] for l in lines])
    # date
    m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", whole)
    if m:
        d = dateparser.parse(m.group(1), settings={"DATE_ORDER":"DMY"})
        dstr = d.strftime("%Y-%m-%d") if d else ""
        # search time nearby
        t = ""
        m2 = re.search(r"(\d{1,2}:\d{2})", whole)
        if m2: t = m2.group(1)
        conf = 0.95 if dstr else 0.5
        return dstr, t, conf, m.group(0)
    # fallback: label search for date/time
    idx, txt, s = fuzzy_label_search(lines, LABEL_MAP["date"]+LABEL_MAP["time"], score_cutoff=45)
    if idx >= 0:
        tln = lines[idx]["text"]
        m3 = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", tln)
        t4 = re.search(r"(\d{1,2}:\d{2})", tln)
        dstr = dateparser.parse(m3.group(1)).strftime("%Y-%m-%d") if m3 else ""
        time = t4.group(1) if t4 else ""
        conf = 0.8 if dstr else 0.4
        return dstr, time, conf, tln
    return "", "", 0.0, ""

def extract_phone(lines: List[Dict[str, Any]]) -> Tuple[str,float,str]:
    whole = " ".join([l["text"] for l in lines])
    m = re.search(r"(\+?\d[\d\-\s]{8,}\d)", whole)
    if m:
        digits = re.sub(r"[^\d]", "", m.group(1))
        if len(digits) == 10: return "+91" + digits, 0.95, m.group(0)
        if len(digits) >= 8: return digits, 0.6, m.group(0)
    # label search
    idx, txt, s = fuzzy_label_search(lines, LABEL_MAP["phone"], score_cutoff=45)
    if idx >= 0:
        m2 = re.search(r"(\+?\d[\d\-\s]{8,}\d)", lines[idx]["text"])
        if m2:
            digits = re.sub(r"[^\d]", "", m2.group(1))
            return ("+91"+digits if len(digits)==10 else digits), 0.85, lines[idx]["text"]
    return "", 0.0, ""

def extract_sections(lines: List[Dict[str, Any]]) -> Tuple[List[str],float,str]:
    idx, txt, s = fuzzy_label_search(lines, LABEL_MAP["sections"], score_cutoff=40)
    found = []
    evidence = ""
    conf = 0.0
    if idx >= 0:
        evidence = lines[idx]["text"]
        nums = re.findall(r"\b(\d{1,3})\b", lines[idx]["text"])
        if nums:
            found = list(dict.fromkeys(nums))
            conf = 0.9
        else:
            # neighbors
            for j in range(idx+1, min(len(lines), idx+4)):
                nums = re.findall(r"\b(\d{1,3})\b", lines[j]["text"])
                if nums: found.extend(nums)
            found = list(dict.fromkeys(found))
            conf = 0.7 if found else 0.3
    else:
        # global fallback numeric extraction (but lower confidence)
        whole = " ".join([l["text"] for l in lines])
        nums = re.findall(r"\b(\d{1,3})\b", whole)
        found = list(dict.fromkeys([n for n in nums if int(n) < 1000]))
        conf = 0.25 if found else 0.0
        evidence = "global-numeric"
    return found, conf, evidence

def extract_police_station(lines: List[Dict[str, Any]]) -> Tuple[str,float,str]:
    idx, txt, s = fuzzy_label_search(lines, LABEL_MAP["ps"], score_cutoff=40)
    if idx >= 0:
        ln = lines[idx]["text"]
        parts = re.split(r":|-", ln, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip(), 0.9, ln
        # else neighbor
        if idx+1 < len(lines):
            return lines[idx+1]["text"], 0.6, lines[idx+1]["text"]
        return ln, 0.5, ln
    # fallback search for "P.S." tokens
    for ln in lines:
        if re.search(r"\bP\.S\.", ln["text"], flags=re.I) or re.search(r"\bPS\b", ln["text"], flags=re.I):
            parts = re.split(r"P\.S\.|PS", ln["text"], flags=re.I)
            if len(parts) > 1 and parts[1].strip(): return parts[1].strip(), 0.7, ln["text"]
    return "", 0.0, ""

def extract_state_district(lines: List[Dict[str, Any]]) -> Tuple[str,str,float,str]:
    whole = " ".join([l["text"] for l in lines])
    # try 'District' label
    m = re.search(r"District[^:\n]*[:\-]?\s*([A-Za-z\u0900-\u097F0-9\-\s,]+)", whole, flags=re.I)
    dist = ""; state = ""; evidence = ""
    if m:
        # split by comma maybe 'District: X, State: Y' or 'District X (State Y)'
        chunk = m.group(1).strip()
        evidence = m.group(0)
        parts = [p.strip() for p in re.split(r",|\(|\)", chunk) if p.strip()]
        if parts:
            dist = parts[0]
            if len(parts) > 1: state = parts[-1]
    # fuzzy-match any candidate to state gazetteer
    if state:
        best = process.extractOne(state, STATE_GAZ, scorer=fuzz.WRatio) or (None,0)
        if best and best[1] > 60: state = best[0]
    else:
        # try global fuzzy
        best = process.extractOne(whole, STATE_GAZ, scorer=fuzz.WRatio) or (None,0)
        if best and best[1] > 60:
            state = best[0]; evidence = "global-state-match"
    conf = 0.6 if dist or state else 0.0
    return dist, state, conf, evidence

def extract_names(lines: List[Dict[str, Any]], full_text: str, use_ner: bool=False) -> Tuple[List[str], float, str]:
    # look for label 'Name' lines
    names = []
    evidence = ""
    idx, txt, s = fuzzy_label_search(lines, LABEL_MAP["name"], score_cutoff=35)
    if idx >= 0:
        ln = lines[idx]["text"]
        parts = re.split(r"[:\-]\s*", ln, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            names.append(parts[1].strip())
            evidence = ln
    # If HF NER available & allowed, apply over full_text to find PERSON entities
    if use_ner and HF_AVAILABLE and hf_ner is not None:
        try:
            ents = hf_ner(full_text[:8000])  # restrict length
            for e in ents:
                label = e.get("entity_group") or e.get("entity")
                if label and label.lower() in ("per","person","personne","name"):
                    txt = e.get("word") or e.get("entity")
                    if txt and txt not in names:
                        names.append(txt)
            if names:
                return names, 0.8, "ner"
        except Exception:
            pass
    if names:
        return names, 0.6, evidence
    return [], 0.0, ""

# -----------------------
# Orchestration
# -----------------------
def extract_from_pdf_bytes(pdf_bytes: bytes, use_ner: bool=False) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text, pages_lines = "", []
    for page in doc:
        txt = page.get_text("text") or ""
        if txt and len(txt.strip())>30:
            # build line-like structures
            lines = []
            for i, l in enumerate(txt.splitlines()):
                if l.strip():
                    lines.append({"text": l.strip(), "bbox": (0,i*10,1000,i*10+8), "y": i*10, "conf": 1.0})
        else:
            lines = ocr_page_with_bboxes(page, dpi=250)
        pages_lines.append(lines)
        full_text += " " + " ".join([ln["text"] for ln in lines])

    # pick best page for metadata: page with most label keyword hits
    best_score = -1; best_idx = 0
    all_label_variants = sum(LABEL_MAP.values(), [])
    for i, pl in enumerate(pages_lines):
        txt = " ".join([ln["text"] for ln in pl]).lower()
        score = sum(1 for v in all_label_variants if v in txt)
        if score > best_score:
            best_score = score; best_idx = i
    if best_score <= 0:
        best_idx = 0
    page_lines = pages_lines[best_idx] if pages_lines else []

    # paragraphs
    paras = paragraphs_from_lines(page_lines, y_gap=12.0)
    evidence_context = "\n".join(paras[:4]) if paras else " ".join([ln["text"] for ln in page_lines[:30]])

    # extract fields
    fir_no, ev_fir = extract_fir_no_from_page(page_lines)
    date_val, time_val, conf_dt, ev_dt = extract_date_time(page_lines)
    phone_val, conf_phone, ev_phone = extract_phone(page_lines)
    sections, conf_sec, ev_sec = extract_sections(page_lines)
    ps, conf_ps, ev_ps = extract_police_station(page_lines)
    dist, state, conf_ds, ev_ds = extract_state_district(page_lines)
    names, conf_names, ev_names = extract_names(page_lines, full_text, use_ner=use_ner)
    address_val, conf_addr, ev_addr = "", 0.0, ""
    # address extraction
    for ln in page_lines:
        if re.search(r"\b(Address|पत्ता)\b", ln["text"], flags=re.I):
            # collect next few lines
            idx = page_lines.index(ln)
            addr = ln["text"]
            for j in range(idx+1, min(idx+4, len(page_lines))):
                addr += " " + page_lines[j]["text"]
            address_val, conf_addr, ev_addr = addr.strip(), 0.6, ln["text"]
            break

    result = {
        "fir_no": fir_no,
        "date": date_val,
        "time": time_val,
        "phone": phone_val,
        "police_station": ps,
        "dist_name": dist,
        "state_name": state,
        "under_sections": sections,
        "under_acts": [],  # acts extraction can be added (similar to sections)
        "names": names,
        "address": address_val,
        "_meta": {
            "best_page_index": best_idx,
            "evidence_context": evidence_context
        },
        "_evidence": {
            "fir_no": ev_fir, "date": ev_dt, "time": ev_dt,
            "phone": ev_phone, "police_station": ev_ps, "state_dist": ev_ds,
            "sections": ev_sec, "names": ev_names, "address": ev_addr
        },
        "_confidence": {
            "fir_no": round(float(0.95 if fir_no else 0.0),2),
            "date": round(float(conf_dt),2),
            "time": round(float(0.8 if time_val else 0.0),2),
            "phone": round(float(conf_phone),2),
            "police_station": round(float(conf_ps),2),
            "state_dist": round(float(conf_ds),2),
            "sections": round(float(conf_sec),2),
            "names": round(float(conf_names),2),
            "address": round(float(conf_addr),2)
        }
    }
    return result

# -----------------------
# Streamlit UI
# -----------------------
st.title("FIR PII Extractor — Best Final (Ensemble)")

st.markdown(
    "Drop a single FIR PDF (Hindi/English). This version uses layout-aware OCR (EasyOCR), "
    "deterministic extractors and optional NER fusion to give robust PII extraction. "
    "If results are still imperfect, correct fields in the editor and download JSON."
)

uploaded = st.file_uploader("Upload FIR PDF", type=["pdf"])
use_ner_flag = st.checkbox("Use optional HF NER fusion (slower, may download model)", value=False) and HF_AVAILABLE

if uploaded:
    pdf_bytes = uploaded.read()
    with st.spinner("Running extraction (OCR + analysis)..."):
        try:
            results = extract_from_pdf_bytes(pdf_bytes, use_ner=use_ner_flag)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            raise

    st.success("Extraction complete — review & correct as needed.")

    st.subheader("Detected evidence (metadata snippet)")
    st.text_area("Evidence context (auto-detected)", results["_meta"]["evidence_context"], height=180)

    st.subheader("Extracted fields (editable)")
    cols = st.columns([3,1,1])
    edited = {}
    with cols[0]:
        edited["fir_no"] = st.text_input("FIR No", value=results.get("fir_no",""))
        edited["date"] = st.text_input("Date (YYYY-MM-DD)", value=results.get("date",""))
        edited["time"] = st.text_input("Time (HH:MM)", value=results.get("time",""))
        edited["police_station"] = st.text_input("Police Station", value=results.get("police_station",""))
        edited["state_name"] = st.text_input("State", value=results.get("state_name",""))
        edited["dist_name"] = st.text_input("District", value=results.get("dist_name",""))
        edited["phone"] = st.text_input("Phone", value=results.get("phone",""))
        edited["address"] = st.text_area("Address", value=results.get("address",""), height=90)
        edited["under_sections"] = st.text_area("Under Sections (one per line)", value="\n".join(results.get("under_sections",[])), height=100)
        edited["names"] = st.text_area("Names (one per line)", value="\n".join(results.get("names",[])), height=100)

    with cols[1]:
        st.subheader("Confidence")
        st.json(results.get("_confidence", {}))
        st.subheader("Evidence snippets")
        st.json(results.get("_evidence", {}))

    with cols[2]:
        st.subheader("Actions")
        if st.button("Auto-normalize phone/date/sections"):
            # phone normalization
            p = edited.get("phone","")
            digits = re.sub(r"[^\d]", "", p)
            if len(digits) == 10: edited["phone"] = "+91"+digits
            # date normalize
            if edited.get("date"):
                d = dateparser.parse(edited["date"], settings={"DATE_ORDER":"DMY"})
                if d: edited["date"] = d.strftime("%Y-%m-%d")
            # sections normalize
            secs = [re.sub(r"[^\d]","",s) for s in edited.get("under_sections","").splitlines()]
            secs = [s for s in secs if s]
            edited["under_sections"] = "\n".join(secs)
            st.success("Normalization done.")

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
                "names": [s for s in edited.get("names","").splitlines() if s.strip()],
                "_meta": results.get("_meta", {}),
                "_evidence": results.get("_evidence", {}),
                "_confidence": results.get("_confidence", {})
            }
            st.download_button("Download final JSON", json.dumps(out, ensure_ascii=False, indent=2),
                               file_name="fir_pii_best.json", mime="application/json")
else:
    st.info("Upload one FIR PDF to extract. For best accuracy: use high-quality scans >= 300 DPI.")
