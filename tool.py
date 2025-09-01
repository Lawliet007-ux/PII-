"""
FIR PII Extractor — Carbon Copy ChatGPT Style
Supports Hindi + English FIRs, from PDF or pasted text.
"""

import streamlit as st
import fitz, pdfplumber, pytesseract
from PIL import Image
import re, os, io, json, base64, tempfile, unicodedata
from rapidfuzz import process
from transformers import pipeline

# ---------- Config ----------
st.set_page_config(page_title="FIR PII Extractor", layout="wide")

# Load multilingual NER (HuggingFace)
@st.cache_resource
def load_ner():
    try:
        return pipeline("ner", model="ai4bharat/indic-ner", aggregation_strategy="simple")
    except Exception:
        return None

NER_PIPE = load_ner()

# Known dictionaries
INDIAN_STATES = ["Maharashtra","Uttar Pradesh","Delhi","Karnataka","Bihar","Gujarat","Tamil Nadu","Madhya Pradesh","Rajasthan","West Bengal"]
DISTRICTS_UP = ["मेरठ","लखनऊ","वाराणसी","कानपुर","नोएडा","गाजियाबाद"]
ACTS = ["IPC 1860","CrPC 1973","Arms Act 1959","NDPS Act 1985","IT Act 2000","Maharashtra Police Act 1951"]

# ---------- Helpers ----------

def clean_text(text: str) -> str:
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    text = re.sub(r'[^\w\s:/().,\\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fix_broken_devanagari(word: str) -> str:
    return re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", word)

def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc: text += page.get_text("text") + "\n"
    except: pass
    if not text.strip():
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t: text += t + "\n"
        except: pass
    if not text.strip():
        try:
            doc = fitz.open(path)
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img, lang="hin+eng") + "\n"
        except: pass
    return clean_text(text)

def fuzzy_match(val, choices):
    if not val: return None
    val = fix_broken_devanagari(val)
    best, score, _ = process.extractOne(val, choices)
    return best if score > 70 else val

# ---------- Extractor ----------

def extract_pii(text: str) -> dict:
    out = {"year": None,"state_name": None,"dist_name": None,"police_station": None,
           "under_acts": None,"under_sections": None,"revised_case_category": None,
           "oparty": None,"name": None,"address": None,"jurisdiction": None,"jurisdiction_type": None}
    
    text = clean_text(text)

    # Year
    m = re.search(r"(19|20)\d{2}", text)
    if m: out["year"] = m.group(0)

    # State (fuzzy)
    for state in INDIAN_STATES:
        if state.lower() in text.lower() or state in text:
            out["state_name"] = state

    # District
    m = re.search(r"(?:District|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]+)", text)
    if m: out["dist_name"] = fuzzy_match(m.group(1), DISTRICTS_UP)

    # Police Station
    m = re.search(r"(?:Police Station|P\.S\.|पोलीस ठाणे)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]+)", text)
    if m: out["police_station"] = fix_broken_devanagari(m.group(1))

    # Acts
    acts = [a for a in ACTS if a.lower().split()[0] in text.lower()]
    out["under_acts"] = acts if acts else None

    # Sections
    secs = re.findall(r"(?:u/s|Section|धारा|कलम)\s*([0-9, ]{1,50})", text, re.IGNORECASE)
    if secs:
        flat = re.findall(r"\d{1,3}", " ".join(secs))
        out["under_sections"] = list(sorted(set(flat)))

    # Revised Case Category
    if out["under_sections"]:
        if any(s in ["354","376","509"] for s in out["under_sections"]):
            out["revised_case_category"] = "SEXUAL_OFFENCE"
        elif "302" in out["under_sections"]:
            out["revised_case_category"] = "MURDER"
        elif "420" in out["under_sections"]:
            out["revised_case_category"] = "FRAUD"
        else:
            out["revised_case_category"] = "OTHER"
    elif out["under_acts"] and "Arms Act" in " ".join(out["under_acts"]):
        out["revised_case_category"] = "WEAPONS"

    # NER extraction
    if NER_PIPE:
        ents = NER_PIPE(text)
        for e in ents:
            if e["entity_group"] == "PER" and not out["name"]:
                out["name"] = e["word"]
                out["oparty"] = "Complainant"
            elif e["entity_group"] == "LOC" and not out["address"]:
                out["address"] = fix_broken_devanagari(e["word"])

    # Cleanup placeholders
    for k in ["name","address","dist_name","police_station"]:
        if out[k] and out[k].strip() in ["नाव","Name","Type","Address"]:
            out[k] = None

    # Jurisdiction
    out["jurisdiction"] = out.get("dist_name") or out.get("state_name")
    out["jurisdiction_type"] = "DISTRICT" if out.get("dist_name") else "STATE"

    return out

# ---------- Streamlit UI ----------
st.title("FIR PII Extractor — ChatGPT-Style Results")
st.write("Upload FIR PDFs or paste text. Extracts clean PII in structured JSON.")

uploaded_files = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
raw_text = st.text_area("Or paste FIR text here", height=200)

if st.button("Extract PII"):
    results = {}
    if uploaded_files:
        for uploaded in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            text = extract_text_from_pdf(tmp_path)
            results[uploaded.name] = extract_pii(text)
            os.remove(tmp_path)
    if raw_text.strip():
        results["pasted_text"] = extract_pii(raw_text)
    if results:
        st.subheader("Extracted PII")
        st.json(results, expanded=True)
        b64 = base64.b64encode(json.dumps(results, ensure_ascii=False, indent=2).encode()).decode()
        st.markdown(f"[Download JSON](data:application/json;base64,{b64})")
    else:
        st.warning("Please upload a PDF or paste text.")
