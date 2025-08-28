# app.py
import streamlit as st
import re
import json
from dateutil import parser as dateparser
from langdetect import detect, DetectorFactory
from rapidfuzz import fuzz
import pandas as pd
from typing import List, Dict, Any, Tuple
import html

DetectorFactory.seed = 0  # deterministic langdetect

st.set_page_config(page_title="PII Extractor (Multilingual Hindi/English)", layout="wide")

# --------------------------
# Helper: OCR normalization
# --------------------------
def normalize_ocr(text: str) -> str:
    """
    Basic OCR cleanup:
    - fix common OCR artifacts (we can expand this list)
    - unify punctuation and whitespace
    """
    if not text:
        return text
    t = text
    # replace weird unicode that OCR often inserts
    t = t.replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '-')
    # common misreads: 0/O, 1/I (be careful — don't auto-fix numerals blindly)
    # remove excessive spaces/newlines
    t = re.sub(r'\r', '\n', t)
    t = re.sub(r'\n{2,}', '\n\n', t)
    t = re.sub(r'[ ]{2,}', ' ', t)
    # remove unexpected nulls
    t = t.replace('\x00', '')
    return t.strip()

# --------------------------
# Regex PII patterns (high-recall)
# --------------------------
REGEX_PATTERNS = {
    # Emails
    "email": re.compile(r'[\w\.-]+@[\w\.-]+\.\w{2,}', re.IGNORECASE),
    # Indian mobile numbers: variants with +91, 0, spaces, hyphens
    "phone": re.compile(r'(?:(?:\+91|91|0)?[\-\s]?)?(?:\d{10}|\d{5}[\-\s]\d{5}|\d{3}[\s]\d{3}[\s]\d{4})'),
    # Dates: many formats dd/mm/yyyy, dd-mm-yyyy, dd month yyyy etc. (will parse later)
    "date": re.compile(r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s*\d{2,4})\b', re.IGNORECASE),
    # Times like 21:33 or 21.33 or 21:09 hours
    "time": re.compile(r'\b(?:[01]?\d|2[0-3])[:\.][0-5]\d(?:\s*hours?)?\b', re.IGNORECASE),
    # PAN (India) — 5 letters, 4 digits, 1 letter
    "pan": re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b', re.IGNORECASE),
    # Aadhaar (may be spaced or hyphenated)
    "aadhaar": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    # Vehicle registration (Indian variety e.g. MH12DE1234 or MH 12 AB 1234)
    "vehicle": re.compile(r'\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,3}[-\s]?\d{1,4}\b', re.IGNORECASE),
    # FIR numbers — heuristic: FIR No., FIR No (with number)
    "fir_no": re.compile(r'\bFIR\s*(?:No\.?|Number|No)\s*[:\-]?\s*[\w\/\-\d]+\b', re.IGNORECASE),
    # Monetary amounts (₹, Rs., INR)
    "amount": re.compile(r'(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?'),
    # Generic ID-like tokens (case: 'GD Ref', 'GD No', 'General Diary Reference', 'G.D. Ref')
    "gd_ref": re.compile(r'\b(?:General Diary Reference|G\.?D\.? Reference|GD Ref|GD No\.?)\s*[:\-]?\s*[\w\d\/\-]+\b', re.IGNORECASE),
    # Postal codes (PIN) India: 6 digits
    "pin": re.compile(r'\b\d{6}\b'),
    # Names (very rough) — capture 'Mr. X', 'Smt.', 'Shri', capitalized sequences (we will validate)
    "honorific_name": re.compile(r'\b(?:Mr|Ms|Mrs|Shri|Smt|Sri|Dr)\.?\s+[A-Z][\w\.\-]+\b'),
}

# --------------------------
# Post-processing / normalization
# --------------------------
def normalize_phone(p: str) -> str:
    s = re.sub(r'[^\d]', '', p)
    if len(s) == 10:
        return '+91' + s
    if len(s) == 12 and s.startswith('91'):
        return '+' + s
    return p  # fallback

def parse_date(text: str) -> Tuple[str, bool]:
    """
    Try to parse a date string; returns (iso_str, success)
    """
    try:
        dt = dateparser.parse(text, dayfirst=True)
        if dt:
            return dt.date().isoformat(), True
    except Exception:
        pass
    return text, False

# --------------------------
# Primary extraction logic
# --------------------------
def extract_with_regex(text: str) -> List[Dict[str, Any]]:
    results = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            span = m.group(0).strip()
            start, end = m.span()
            normalized = span
            confidence = 0.75  # baseline for regex
            # small normalizations
            if label == "phone":
                normalized = normalize_phone(span)
                # phone -> increase confidence if length looks correct
                digits = re.sub(r'\D', '', span)
                if len(digits) in (10, 12):
                    confidence = 0.92
                else:
                    confidence = 0.6
            if label == "date":
                parsed, ok = parse_date(span)
                if ok:
                    normalized = parsed
                    confidence = 0.9
                else:
                    confidence = 0.6
            if label in ("aadhaar", "pan", "pin"):
                confidence = 0.95
            if label == "email":
                confidence = 0.98
            results.append({
                "label": label,
                "text": span,
                "normalized": normalized,
                "start": start,
                "end": end,
                "confidence": confidence
            })
    return results

# --------------------------
# Optional ML NER hooks (Stanza / spaCy)
# --------------------------
# NOTE: we don't import stanza/spacy by default to avoid forcing downloads.
def ml_ner_stub(text: str, lang_hint: str) -> List[Dict[str, Any]]:
    """
    Hook for integrating ML NER (Stanza or spaCy). This function intentionally
    does nothing by default. To enable:
      - pip install stanza
      - import stanza
      - stanza.download('hi'); stanza.download('en')
      - initialize pipelines outside this function
      - replace the body with calls to stanza or spacy and return
        list of dicts with keys: label, text, start, end, confidence, normalized
    """
    return []


# --------------------------
# Merge + deduplicate results
# --------------------------
def merge_results(results: List[Dict[str, Any]], dedup_threshold: float = 90.0) -> List[Dict[str, Any]]:
    """
    Merge results that are near-duplicates using fuzzy matching (to avoid duplicates from regex + NER).
    We keep the highest-confidence entry for duplicates.
    """
    merged = []
    for r in results:
        placed = False
        for m in merged:
            # if same label and fuzzy match high OR spans overlap -> merge
            if r['label'] == m['label']:
                # overlap check
                overlap = not (r['end'] < m['start'] or r['start'] > m['end'])
                ratio = fuzz.partial_ratio(r['text'], m['text'])
                if overlap or ratio >= dedup_threshold:
                    # keep the one with higher confidence (and longer normalized text)
                    if r['confidence'] > m['confidence']:
                        m.update(r)
                    placed = True
                    break
        if not placed:
            merged.append(r.copy())
    # sort by start position
    merged.sort(key=lambda x: x.get('start', 0))
    return merged

# --------------------------
# Top-level extraction
# --------------------------
def extract_pii_from_text(text: str, enable_ml: bool = False) -> Dict[str, Any]:
    text = normalize_ocr(text)
    # try language detection (but may fail on very small text)
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    # regex extraction (fast, offline)
    regex_hits = extract_with_regex(text)
    # optional ML
    ml_hits = []
    if enable_ml:
        ml_hits = ml_ner_stub(text, lang)
    combined = regex_hits + ml_hits
    merged = merge_results(combined)
    # build structured output per label
    out = {}
    for r in merged:
        out.setdefault(r['label'], []).append({
            "text": r['text'],
            "normalized": r.get('normalized', r['text']),
            "confidence": r.get('confidence', 0.0),
            "span": [r['start'], r['end']]
        })
    # meta
    return {
        "language_hint": lang,
        "raw_text": text,
        "counts": {k: len(v) for k, v in out.items()},
        "entities": out
    }

# --------------------------
# Streamlit UI
# --------------------------
st.title("PII Extractor — Multilingual (Hindi / English)")

with st.sidebar:
    st.markdown("### Input options")
    input_mode = st.radio("Input type", ["Paste text", "Upload file (txt/json)", "Paste Elasticsearch JSON"])
    enable_ml = st.checkbox("Enable ML NER hook (Stanza/spaCy) — requires additional setup", value=False)
    st.markdown("**Extraction settings**")
    dedup_thresh = st.slider("Dedup fuzzy threshold", 75, 100, 90)
    st.markdown("---")
    st.markdown("**Notes**")
    st.write("This tool uses regex by default. For best accuracy on noisy/legal OCR, enable an ML NER model (see README in code comments).")

text_input = ""
if input_mode == "Paste text":
    text_input = st.text_area("Paste the OCR / Elasticsearch text here", height=300)
elif input_mode == "Upload file (txt/json)":
    uploaded = st.file_uploader("Upload text or JSON file", type=["txt", "json"])
    if uploaded:
        raw = uploaded.read()
        try:
            # decode bytes robustly
            text_input = raw.decode("utf-8")
        except:
            try:
                text_input = raw.decode("latin-1")
            except:
                text_input = str(raw)
        # if JSON, try to extract fields
        if uploaded.type == "application/json" or uploaded.name.endswith(".json"):
            try:
                j = json.loads(text_input)
                # Try to extract typical Elasticsearch fields: _source, hits, _source.text, message
                # Flatten heuristically
                candidates = []
                def walk(obj):
                    if isinstance(obj, dict):
                        for k,v in obj.items():
                            if isinstance(v, (dict, list)):
                                walk(v)
                            else:
                                if isinstance(v, str) and len(v) > 10:
                                    candidates.append(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            walk(item)
                walk(j)
                if candidates:
                    text_input = "\n\n".join(candidates)
            except Exception:
                pass
elif input_mode == "Paste Elasticsearch JSON":
    text_input = st.text_area("Paste ES JSON response (we will try to extract text fields)", height=300)

if st.button("Extract PII"):
    if not text_input or text_input.strip() == "":
        st.error("Please provide text or a file to extract from.")
    else:
        with st.spinner("Extracting..."):
            try:
                result = extract_pii_from_text(text_input, enable_ml=enable_ml)
                # apply dedup threshold selection: re-merge with user threshold
                # (re-run merge step)
                all_entities = []
                # flatten for merging
                for label, arr in result['entities'].items():
                    for ent in arr:
                        all_entities.append({
                            "label": label,
                            "text": ent['text'],
                            "normalized": ent['normalized'],
                            "start": ent['span'][0],
                            "end": ent['span'][1],
                            "confidence": ent['confidence']
                        })
                merged = merge_results(all_entities, dedup_threshold=dedup_thresh)
                # rebuild results map
                out_map = {}
                for m in merged:
                    out_map.setdefault(m['label'], []).append({
                        "text": m['text'],
                        "normalized": m.get('normalized', m['text']),
                        "confidence": m.get('confidence', 0.0)
                    })
                result['entities'] = out_map

                st.success("Extraction complete")
                col1, col2 = st.columns([2,1])
                with col1:
                    st.subheader("Detected entities")
                    # Display table of all extracted entities
                    rows = []
                    for label, arr in result['entities'].items():
                        for ent in arr:
                            rows.append({
                                "label": label,
                                "text": ent['text'],
                                "normalized": ent['normalized'],
                                "confidence": ent['confidence']
                            })
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df)
                        st.markdown("**Download results**")
                        st.download_button("Download JSON", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="pii_extraction.json", mime="application/json")
                        st.download_button("Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="pii_extraction.csv", mime="text/csv")
                    else:
                        st.write("No entities detected.")
                with col2:
                    st.subheader("Summary")
                    st.write(f"Language hint: **{result['language_hint']}**")
                    st.write("Counts:")
                    for k,v in result['counts'].items():
                        st.write(f"- {k}: {v}")
                    st.subheader("Raw text (cleaned)")
                    st.code(result['raw_text'][:2000] + ("..." if len(result['raw_text'])>2000 else ""))
            except Exception as e:
                st.error(f"Extraction failed: {e}")

st.markdown("---")
st.markdown("### Example / Sample usage")
st.markdown("Paste the sample you provided in the prompt into the text area and click Extract PII. Expand regex map as needed for local/region specifics.")
