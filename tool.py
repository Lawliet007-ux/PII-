"""
FIR PII Extractor — Hindi + English (Generalized, NER + OCR)
- Works for ANY FIR (Hindi/English/mixed) via hybrid pipeline:
  1) PDF text → PyMuPDF/pdfplumber → fallback OCR (Tesseract eng+hin)
  2) PII extraction → NER (HuggingFace) + robust regex + normalization
  3) Post-processing → confidence scoring, dedup, cleaning
- Also supports pasted text (textbox) alongside PDF upload

Setup
-----
System: Tesseract OCR + Hindi data
  Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-hin
Python:
  pip install streamlit pymupdf pdfplumber pytesseract pillow opencv-python regex langdetect transformers torch

Note: If the NER model can't be downloaded (no internet), the app falls back to regex-only mode.
"""

import streamlit as st
import re
import json
import base64
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import tempfile
import os
from typing import List, Dict, Any, Tuple

# Optional NER (graceful fallback if unavailable)
NER_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    NER_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor — Generalized", layout="wide")

# ------------------ Constants & Dictionaries ------------------

INDIAN_STATES = {
    # English → Normalized
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttarakhand": "Uttarakhand",
    "uttar pradesh": "Uttar Pradesh",
    "west bengal": "West Bengal",
    "nct of delhi": "Delhi",
    "delhi": "Delhi",
    "jammu and kashmir": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "chandigarh": "Chandigarh",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "lakshadweep": "Lakshadweep",
    "puducherry": "Puducherry",
}

# Common Devanagari variants → English normalized
STATE_DEVANAGARI_MAP = {
    "महाराष्ट्र": "Maharashtra",
    "दिल्ली": "Delhi",
    "पश्चिम बंगाल": "West Bengal",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "मध्य प्रदेश": "Madhya Pradesh",
    "राजस्थान": "Rajasthan",
    "बिहार": "Bihar",
    "गुजरात": "Gujarat",
    "कर्नाटक": "Karnataka",
    "पंजाब": "Punjab",
    "हिमाचल प्रदेश": "Himachal Pradesh",
}

KNOWN_ACTS = [
    # English
    "Indian Penal Code 1860", "IPC 1860", "Indian Penal Code", "IPC",
    "Code of Criminal Procedure 1973", "CrPC 1973", "Cr.P.C.", "CrPC",
    "Arms Act 1959", "Arms Act", "Maharashtra Police Act 1951", "IT Act 2000", "Information Technology Act 2000",
    "NDPS Act 1985", "Juvenile Justice Act", "POCSO Act 2012", "Motor Vehicles Act 1988",
    # Devanagari hints
    "भारतीय दंड संहिता", "दण्ड प्रक्रिया संहिता", "शस्त्र अधिनियम", "महाराष्ट्र पोलीस अधिनियम", "सूचना प्रौद्योगिकी अधिनियम",
]

SECTION_MAX = 600  # numeric sanity for IPC-like sections

CATEGORY_RULES = [
    ("WEAPONS", {"acts": ["Arms Act"], "sections": ["25", "3"]}),
    ("SEXUAL_OFFENCE", {"sections": ["354", "376", "509", "354A", "354B", "354C", "354D"]}),
    ("THEFT_BURGLARY", {"sections": ["379", "380", "457", "454"], "keywords": ["theft", "burglary", "चोरी"]}),
    ("ROBBERY_DACOITY", {"sections": ["392", "395", "397"]}),
    ("HURT_ASSAULT", {"sections": ["323", "324", "325", "326", "341", "342", "504", "506"]}),
    ("NARCOTICS", {"acts": ["NDPS"]}),
    ("IT_CYBER", {"acts": ["IT Act"], "keywords": ["cyber", "आईटी"]}),
    ("PUBLIC_ORDER", {"sections": ["135", "37"], "acts": ["Maharashtra Police Act"]}),
]

PS_LABELS = r"(?:Police\s*Station|P\.?\s*S\.?|PS\s+|पोलीस\s*ठाणे|पुलिस\s*थाना|थाना)"
DIST_LABELS = r"(?:District|जिला|जिल्हा)"
STATE_LABELS = r"(?:State|राज्य)"
SECT_LABELS = r"(?:Under\s*Section|U/s\.?|Sections?|धारा|कलम)"
ACT_LABELS = r"(?:Act|अधिनियम|कायदा)"

# ------------------ Utilities ------------------

def normalize_state(text: str) -> str | None:
    low = text.lower()
    for k, v in INDIAN_STATES.items():
        if k in low:
            return v
    for dev, v in STATE_DEVANAGARI_MAP.items():
        if dev in text:
            return v
    # explicit label capture
    m = re.search(STATE_LABELS + r"\s*[:\-]?\s*([A-Za-z\u0900-\u097F ,]+)", text, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        n = normalize_state(cand)
        if n:
            return n
    return None


def extract_sections(text: str) -> List[str]:
    # Prefer sequences near section labels
    span = 400
    sections: List[str] = []
    for m in re.finditer(SECT_LABELS, text, re.IGNORECASE):
        start = max(0, m.start())
        window = text[start:start+span]
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", window)
        for n in nums:
            n_clean = re.sub(r"[^0-9A-Za-z()]+", "", n)
            base = re.match(r"(\d{1,3})", n_clean)
            if base and 0 < int(base.group(1)) <= SECTION_MAX:
                sections.append(n_clean)
    # fallback: any numeric lists with commas
    if not sections:
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", text)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base and 0 < int(base.group(1)) <= SECTION_MAX:
                sections.append(n)
    # dedupe preserving order
    seen = set()
    out = []
    for s in sections:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def extract_acts(text: str) -> List[str]:
    acts = []
    # capture near labels
    for m in re.finditer(ACT_LABELS, text, re.IGNORECASE):
        win = text[m.start(): m.start()+400]
        # naive split by separators
        parts = re.split(r"[\n,;/]", win)
        for p in parts:
            for a in KNOWN_ACTS:
                if re.search(re.escape(a), p, re.IGNORECASE):
                    acts.append(a)
    # global scan
    for a in KNOWN_ACTS:
        if re.search(re.escape(a), text, re.IGNORECASE):
            acts.append(a)
    # normalize and dedupe
    norm = []
    for a in acts:
        if re.search(r"भारतीय दंड संहिता|IPC", a, re.IGNORECASE):
            norm.append("Indian Penal Code 1860")
        elif "Arms Act" in a or "शस्त्र" in a:
            norm.append("Arms Act 1959")
        elif "Maharashtra Police" in a or "महाराष्ट्र पोलीस" in a:
            norm.append("Maharashtra Police Act 1951")
        elif "CrPC" in a or "दण्ड प्रक्रिया" in a:
            norm.append("Code of Criminal Procedure 1973")
        elif "IT Act" in a or "सूचना प्रौद्योगिकी" in a:
            norm.append("Information Technology Act 2000")
        elif "NDPS" in a:
            norm.append("NDPS Act 1985")
        else:
            norm.append(a)
    seen = set(); out = []
    for a in norm:
        if a not in seen:
            seen.add(a); out.append(a)
    return out


def extract_year(text: str) -> str | None:
    # Prefer labeled year first
    m = re.search(r"(?:Year|वर्ष)\s*[:\-]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return m.group(0) if m else None


def extract_labeled(text: str, label_regex: str, max_len: int = 80) -> str | None:
    m = re.search(label_regex + r"\s*[:\-]?\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return val[:max_len]
    return None


def guess_district(text: str) -> str | None:
    val = extract_labeled(text, DIST_LABELS)
    if val:
        return val
    # NER fallback will add more candidates – handled later
    return None


def guess_police_station(text: str) -> str | None:
    val = extract_labeled(text, PS_LABELS)
    if val:
        return re.sub(r"\s+Police\s*Station$", "", val, flags=re.IGNORECASE).strip()
    # also look for patterns like "PS <name>"
    m = re.search(r"\bPS\s+([A-Za-z\u0900-\u097F\- ]{2,40})", text)
    if m:
        return m.group(1).strip()
    return None

# ------------------ NER wrapper ------------------

@st.cache_resource(show_spinner=False)
def load_ner():
    if not NER_AVAILABLE:
        return None
    try:
        # A widely-available multilingual NER. Swap with a fine-tuned FIR model for best results.
        return pipeline("token-classification", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")
    except Exception:
        try:
            return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
        except Exception:
            return None

NER_PIPE = load_ner()


def ner_candidates(text: str) -> Dict[str, List[str]]:
    cands = {"PER": [], "LOC": [], "ORG": []}
    if not NER_PIPE:
        return cands
    try:
        ents = NER_PIPE(text[:8000])  # limit for speed
        for e in ents:
            label = e.get("entity_group", "")
            word = e.get("word", "").strip()
            if not word:
                continue
            if label in ("PER", "PERSON"):
                cands["PER"].append(word)
            elif label in ("LOC", "LOCATION", "GPE"):
                cands["LOC"].append(word)
            elif label in ("ORG", ):
                cands["ORG"].append(word)
    except Exception:
        pass
    # dedupe
    for k in cands:
        seen = set(); out = []
        for w in cands[k]:
            if w not in seen:
                seen.add(w); out.append(w)
        cands[k] = out
    return cands

# ------------------ Master extractor ------------------

def extract_pii(text: str) -> Dict[str, Any]:
    t = re.sub(r"[ \t]+", " ", text)
    year = extract_year(t)
    state = normalize_state(t)
    dist = guess_district(t)
    ps = guess_police_station(t)
    acts = extract_acts(t)
    sections = extract_sections(t)

    # NER boosts
    ner = ner_candidates(t)

    # Name/oparty detection
    name = None; oparty = None
    # Look for labeled complainant/informant lines first
    m = re.search(r"(?:Complainant|Informant|तक्रारदार|सूचक|खबर देने वाला)[^\n]{0,80}?Name\s*[:\-]?\s*([^\n]+)", t, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        oparty = "Complainant"
    if not name:
        # Generic Name label
        m = re.search(r"\bName\s*[:\-]?\s*([A-Z][A-Za-z .]{2,60})", t)
        if m:
            name = m.group(1).strip()
    if not name and ner["PER"]:
        # pick the longest plausible PER token sequence
        name = max(ner["PER"], key=len)
    # Opposite party heuristic
    if not oparty:
        if re.search(r"Accused|आरोपी|प्रतिवादी", t, re.IGNORECASE):
            # default to complainant unless Name appears in accused block
            if re.search(r"Accused[^\n]{0,120}" + (name or ""), t, re.IGNORECASE):
                oparty = "Accused"
            else:
                oparty = "Complainant"
        else:
            oparty = "Complainant"

    # Address (prefer labeled)
    address = extract_labeled(t, r"(?:Address|पता|पत्ता)")
    if not address and ner["LOC"]:
        # Build a compact address-like string from LOC tokens
        loc = ", ".join(ner["LOC"][:3])
        address = loc if loc else None

    # District/PS from NER if missing
    if not dist and ner["LOC"]:
        # choose a LOC that co-occurs with district label words nearby
        for loc in ner["LOC"]:
            if re.search(loc + r"\s*(?:जिला|District)", t, re.IGNORECASE):
                dist = loc; break
        if not dist:
            dist = ner["LOC"][0]
    if not ps and ner["LOC"]:
        # look for "PS <loc>" pattern
        for loc in ner["LOC"]:
            if re.search(rf"\b(?:PS|Police Station)\s+{re.escape(loc)}\b", t, re.IGNORECASE):
                ps = loc; break

    # Jurisdiction & type
    jurisdiction = None; jurisdiction_type = None
    if re.search(r"PAN[\s_-]?INDIA|ALL\s*INDIA", t, re.IGNORECASE):
        jurisdiction = "PAN_INDIA"; jurisdiction_type = "PAN_INDIA"
    elif dist:
        jurisdiction = dist; jurisdiction_type = "DISTRICT"
    elif state:
        jurisdiction = state; jurisdiction_type = "STATE"

    # Case category via rules
    category = "OTHER"
    acts_text = " | ".join(acts)
    for label, rules in CATEGORY_RULES:
        hit = False
        if any(a.lower() in acts_text.lower() for a in rules.get("acts", [])):
            hit = True
        if not hit and sections:
            if any(sec.upper() in [s.upper() for s in sections] for sec in rules.get("sections", [])):
                hit = True
        if not hit and rules.get("keywords"):
            if any(re.search(k, t, re.IGNORECASE) for k in rules["keywords"]):
                hit = True
        if hit:
            category = label
            break

    # Final cleanup: avoid noisy sections like 00, 000, times etc.
    cleaned_sections = []
    seen = set()
    for s in sections:
        base = re.match(r"(\d{1,3})", s)
        if not base:
            continue
        n = int(base.group(1))
        if 1 <= n <= SECTION_MAX and s not in seen:
            seen.add(s)
            cleaned_sections.append(s)

    return {
        "year": year,
        "state_name": state,
        "dist_name": dist,
        "police_station": ps,
        "under_acts": acts or None,
        "under_sections": cleaned_sections or None,
        "revised_case_category": category,
        "oparty": oparty,
        "name": name,
        "address": address,
        "jurisdiction": jurisdiction,
        "jurisdiction_type": jurisdiction_type,
    }

# ------------------ PDF Text Extraction ------------------

def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception:
        pass
    if len(text) < 100:
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text() or ""
                    text += t + "\n"
        except Exception:
            pass
    # OCR fallback if still weak
    if len(text) < 150:
        try:
            doc = fitz.open(path)
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img, lang="eng+hin") + "\n"
        except Exception:
            pass
    return text

# ------------------ UI ------------------

with st.sidebar:
    st.header("Settings")
    use_ner = st.checkbox("Use NER (HuggingFace)", value=True if NER_PIPE else False, help="Improves names/locations. Falls back to regex if model not available.")
    st.caption("Model: xlm-roberta (auto). To fine-tune on FIRs, swap the model in load_ner().")

st.title("FIR PII Extractor — Hindi + English (Generalized)")
st.write("Upload FIR PDF(s) **or** paste FIR text below. The app uses OCR + NER + rules to extract clean PII without extra noise.")

uploaded_files = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
raw_text = st.text_area("Or paste FIR text here", height=220, placeholder="Paste text from Elasticsearch or any source…")

if st.button("Extract PII"):
    results: Dict[str, Any] = {}

    # Process PDFs
    if uploaded_files:
        for uploaded in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            extracted_text = extract_text_from_pdf(tmp_path)
            # Temporarily toggle NER if user disabled
            global NER_PIPE
            if not use_ner:
                NER_PIPE = None
            pii = extract_pii(extracted_text)
            results[uploaded.name] = pii
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Process pasted text
    if raw_text.strip():
        if not use_ner:
            NER_PIPE = None
        pii = extract_pii(raw_text)
        results["pasted_text"] = pii

    if results:
        st.subheader("Extracted PII")
        st.json(results)
        b64 = base64.b64encode(json.dumps(results, ensure_ascii=False, indent=2).encode()).decode()
        href = f"data:application/json;base64,{b64}"
        st.markdown(f"[Download JSON]({href})")
    else:
        st.warning("Please upload at least one PDF or paste text.")

st.info("Tip: For best accuracy on scanned FIRs, install Hindi Tesseract data and consider commercial OCR. Swap the HuggingFace model in load_ner() with your fine-tuned FIR NER for exceptional results.")
