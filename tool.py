# fir_pii_extractor_final.py
"""
FIR PII Extractor — Final hybrid pipeline (best-effort ChatGPT-like extraction)
- PDF/text input
- PyMuPDF -> pdfplumber -> Tesseract OCR fallback
- Aggressive OCR cleanup + Devanagari repair
- Regex-first extraction, NER fallback, fuzzy-repair, normalization & inference
- Outputs canonical JSON with the requested fields.
"""

import streamlit as st
import fitz, pdfplumber, pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64
from typing import List, Dict, Any, Optional
from rapidfuzz import process, fuzz

# transformers NER is optional; app will run without it
try:
    from transformers import pipeline
    NER_AVAILABLE = True
except Exception:
    pipeline = None
    NER_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor — Final", layout="wide")

# ------------------ Small canonical dictionaries (extendable) ------------------

# States (normalized english & devanagari hints)
STATES_CANONICAL = {
    "maharashtra": "Maharashtra", "mumbai": "Maharashtra", "pune": "Maharashtra",
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "delhi": "Delhi", "karnataka": "Karnataka", "gujarat": "Gujarat",
    "bihar": "Bihar", "tamil nadu": "Tamil Nadu", "west bengal": "West Bengal",
    "rajasthan": "Rajasthan", "telangana": "Telangana", "andhra pradesh": "Andhra Pradesh",
}
DEVANAGARI_TO_STATE = {
    "महाराष्ट्र": "Maharashtra", "पुणे": "Maharashtra", "पुणे शहर": "Maharashtra",
    "दिल्ली": "Delhi", "उत्तर प्रदेश": "Uttar Pradesh", "कर्नाटक": "Karnataka",
    "गुजरात": "Gujarat", "बिहार": "Bihar", "तमिलनाडु": "Tamil Nadu",
}

# Small sample district lists — add more as you collect FIRs (used for fuzzy repair)
COMMON_DISTRICTS = [
    "Pune", "Pune City", "Mumbai", "Mumbai City", "Nagpur", "Nashik",
    "Meerut", "Lucknow", "Varanasi", "Kanpur", "Noida", "Ghaziabad",
    "Bengaluru", "Bangalore", "Ahmedabad", "Jaipur", "Patna"
]
COMMON_DISTRICTS_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","मेरठ","लखनऊ","वाराणसी","कानपुर"]

# Example police-station names (small seed)
COMMON_POLICE_STATIONS = [
    "Bhosari", "Hadapsar", "Cyber Crime Cell", "Dadar", "Andheri", "Colaba", "Pune City PS"
]
COMMON_POLICE_STATIONS_DEV = ["भोसरी","हडपसर","डादर","अंधेरी","कॉलाबा"]

# Known Acts mapping
KNOWN_ACTS = {
    "ipc": "Indian Penal Code 1860",
    "indian penal code": "Indian Penal Code 1860",
    "crpc": "Code of Criminal Procedure 1973",
    "arms act": "Arms Act 1959",
    "information technology": "Information Technology Act 2000",
    "it act": "Information Technology Act 2000",
    "ndps": "NDPS Act 1985",
    "pocso": "POCSO Act 2012",
    "maharashtra police": "Maharashtra Police Act 1951"
}

SECTION_MAX = 600

# ------------------ Helpers: text cleaning & devnagari repair ------------------

def remove_control_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

def collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()

def strip_odd_unicode(text: str) -> str:
    # keep ASCII and Devanagari block and common punctuation
    text = re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—]", " ", text)
    return text

def clean_text_for_processing(text: str) -> str:
    if not text:
        return ""
    t = remove_control_chars(text)
    t = strip_odd_unicode(t)
    t = collapse_whitespace(t)
    return t

def fix_broken_devanagari_runs(text: str) -> str:
    # Collapse spaces between Devanagari characters repeatedly
    def collapse_once(s):
        return re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", s)
    prev = None
    cur = text
    # iterate until fixed (small loop)
    for _ in range(5):
        prev = cur
        cur = collapse_once(cur)
        if cur == prev:
            break
    return cur

def canonicalize_text(text: str) -> str:
    t = clean_text_for_processing(text)
    t = fix_broken_devanagari_runs(t)
    return t

# ------------------ Text extraction (pdf -> text) ------------------

def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        out = []
        for p in doc:
            out.append(p.get_text("text") or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_pdfplumber(path: str) -> str:
    try:
        out = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                out.append(p.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_ocr(path: str, tesseract_langs: str = "eng+hin+mar") -> str:
    try:
        doc = fitz.open(path)
        out = []
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            txt = pytesseract.image_to_string(img, lang=tesseract_langs)
            out.append(txt or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf(path: str, tesseract_langs: str = "eng+hin+mar") -> str:
    # try in order: PyMuPDF, pdfplumber, OCR
    text = extract_text_pymupdf(path)
    text = canonicalize_text(text)
    if len(text) < 200:
        alt = extract_text_pdfplumber(path)
        alt = canonicalize_text(alt)
        if len(alt) > len(text):
            text = alt
    if len(text) < 200:
        ocr = extract_text_ocr(path, tesseract_langs=tesseract_langs)
        text = canonicalize_text(ocr) if len(canonicalize_text(ocr)) > len(text) else text
    return canonicalize_text(text)

# ------------------ NER pipeline loader (optional, cached) ------------------

@st.cache_resource
def load_ner_pipeline():
    if not NER_AVAILABLE:
        return None
    try:
        # try Indic NER if available, else fallback to a multilingual NER
        try:
            return pipeline("token-classification", model="ai4bharat/indic-ner", aggregation_strategy="simple")
        except Exception:
            return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
    except Exception:
        return None

# ------------------ Field-specific extraction helpers ------------------

def extract_year(text: str) -> Optional[str]:
    # prefer labelled Year / Date patterns then general 4-digit
    m = re.search(r"(?:Year|वर्ष|Date of FIR|Date)\s*[:\-\s]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    return m2.group(0) if m2 else None

def extract_fir_no(text: str) -> Optional[str]:
    m = re.search(r"(?:FIR No|F\\.I\\.R\\.|F\\.I\\.R|FIR|F\\.I\\.R\\. No|F\\.I\\.R No|म. खब)\s*[:\-\s]?\s*([A-Za-z0-9\-\/]{3,50})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def extract_district(text: str) -> Optional[str]:
    # labelled district
    m = re.search(r"(?:District|Dist\.|जिला|जिल्हा)\s*[:\-\s]?\s*([A-Za-z\u0900-\u097F0-9 ,./-]{2,80})", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return val
    # sometimes 'District (' label followed by Dev text
    m2 = re.search(r"District\s*\([^)]+\)\s*[:\-\s]?\s*([A-Za-z\u0900-\u097F ]{2,80})", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def extract_police_station(text: str) -> Optional[str]:
    m = re.search(r"(?:Police Station|P\.S\.|PS\b|पोलीस ठाणे|पुलिस थाना|थाना)\s*[:\-\s]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 'PS <name>' pattern
    m2 = re.search(r"\bPS\s+([A-Za-z\u0900-\u097F\- ]{2,60})", text)
    if m2:
        return m2.group(1).strip()
    return None

def extract_acts(text: str) -> Optional[List[str]]:
    found = []
    # search for known act keywords
    for k in KNOWN_ACTS.keys():
        if re.search(re.escape(k), text, re.IGNORECASE):
            found.append(KNOWN_ACTS[k])
    # search label vicinity for 'Act' words
    for m in re.finditer(r"(?:Act|अधिनियम|कायदा|अधि)\s*[:\-\s]?[^\n]{0,120}", text, re.IGNORECASE):
        chunk = m.group(0)
        for k in KNOWN_ACTS.keys():
            if re.search(re.escape(k), chunk, re.IGNORECASE):
                found.append(KNOWN_ACTS[k])
    return dedupe_preserve_order(found) if found else None

def extract_sections(text: str) -> Optional[List[str]]:
    secs = []
    # scan near section labels
    for m in re.finditer(r"(?:Section|Sections|U/s|U/s\.|धारा|कलम|Sect)\b", text, re.IGNORECASE):
        window = text[m.start(): m.start()+300]
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                v = int(base.group(1))
                if 1 <= v <= SECTION_MAX:
                    secs.append(n)
    # fallback: any numeric sequences that look like sections
    if not secs:
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", text)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                v = int(base.group(1))
                if 1 <= v <= SECTION_MAX:
                    secs.append(n)
    secs = dedupe_preserve_order(secs)
    return secs if secs else None

def extract_name_labelled(text: str) -> Optional[str]:
    # labeled 'Name' fields in English/Hindi/Marathi
    m = re.search(r"(?:Complainant|Informant|Complainant\/Informant|तक्रारदार|सूचक|नाव|Name|नाम)\s*[:\-\s]?\s*([A-Za-z\u0900-\u097F .]{2,120})", text, re.IGNORECASE)
    if m:
        cand = m.group(1).split("\n")[0].strip()
        if cand and cand.lower() not in ("name", "नाव", "नाम"):
            return cand
    # fallback labelled 'Name:' general
    m2 = re.search(r"\bName\s*[:\-\s]\s*([A-Z][A-Za-z .]{2,120})", text)
    if m2:
        return m2.group(1).strip()
    return None

def extract_address_labelled(text: str) -> Optional[str]:
    m = re.search(r"(?:Address|पत्ता|पत्ता[:\-\s]?)\s*[:\-\s]?\s*([A-Za-z0-9\u0900-\u097F,./\- ]{6,250})", text, re.IGNORECASE)
    if m:
        addr = m.group(1).strip()
        # cut at phone/mobile labels
        addr = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Passport)", addr, flags=re.IGNORECASE)[0].strip()
        return addr
    return None

# ------------------ small utilities ------------------

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for i in items:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def fuzzy_repair(candidate: str, choices: List[str], threshold: int = 70) -> str:
    if not candidate or not choices:
        return candidate
    cand = candidate.strip()
    # collapse broken devanagari
    cand = fix_broken_devanagari_runs(cand)
    best = process.extractOne(cand, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return cand

# ------------------ Category mapping ------------------

def map_case_category(acts: Optional[List[str]], sections: Optional[List[str]], text: str) -> str:
    # heavy-priority mapping
    if acts:
        acts_text = " ".join([a.lower() for a in acts])
        if "information technology" in acts_text or "it act" in acts_text:
            return "CYBER_CRIME"
        if "arms act" in acts_text:
            return "WEAPONS"
        if "ndps" in acts_text:
            return "NARCOTICS"
    if sections:
        sset = set([re.match(r"(\d{1,3})", s).group(1) for s in sections if re.match(r"(\d{1,3})", s)])
        if any(x in sset for x in ("354","376","509")):
            return "SEXUAL_OFFENCE"
        if "302" in sset:
            return "MURDER"
        if any(x in sset for x in ("420","406")):
            return "FRAUD"
        if any(x in sset for x in ("323","325","326")):
            return "HURT_ASSAULT"
    # keywords fallback
    if re.search(r"\bc y b e r\b|\bcyber\b|कायदा|हॅक|हैक", text, re.IGNORECASE):
        return "CYBER_CRIME"
    return "OTHER"

# ------------------ NER-based boosters ------------------

def ner_boost(text: str, ner_pipe) -> Dict[str, List[str]]:
    res = {"PER": [], "LOC": [], "ORG": []}
    if ner_pipe is None:
        return res
    try:
        ents = ner_pipe(text[:8000])
        for e in ents:
            grp = e.get("entity_group") or e.get("entity")
            w = e.get("word") or e.get("word") or ""
            w = w.strip()
            if not w:
                continue
            if grp in ("PER","PERSON"):
                res["PER"].append(w)
            elif grp in ("LOC","LOCATION","GPE"):
                res["LOC"].append(w)
            elif grp in ("ORG",):
                res["ORG"].append(w)
    except Exception:
        pass
    for k in res:
        res[k] = dedupe_preserve_order(res[k])
    return res

# ------------------ Master extractor ------------------

def extract_pii_full(text: str, use_ner: bool, ner_pipe) -> Dict[str, Any]:
    t = canonicalize_text(text)

    # Layer 1: strict regex / labelled extraction
    year = extract_year(t)
    fir_no = extract_fir_no(t)
    district = extract_district(t)
    police_station = extract_police_station(t)
    acts = extract_acts(t)
    sections = extract_sections(t)
    name = extract_name_labelled(t)
    address = extract_address_labelled(t)

    # Layer 2: NER fallback / boost
    per_candidates = []; loc_candidates = []
    if use_ner and ner_pipe:
        ents = ner_boost(t, ner_pipe)
        per_candidates = ents.get("PER", [])
        loc_candidates = ents.get("LOC", [])

    # if no labelled name, take NER person
    if not name and per_candidates:
        name = per_candidates[0]

    # if district missing, try NER locs or fuzzy repair against dev list
    if not district and loc_candidates:
        district = loc_candidates[0]

    # Post-process: repair broken Devanagari & remove placeholders
    if district:
        district = fix_broken_devanagari_runs(district)
        # fuzzy repair to canonical lists (Dev + Eng)
        district = fuzzy_repair(district, COMMON_DISTRICTS + COMMON_DISTRICTS_DEV, threshold=65)
    if police_station:
        police_station = fix_broken_devanagari_runs(police_station)
        police_station = fuzzy_repair(police_station, COMMON_POLICE_STATIONS + COMMON_POLICE_STATIONS_DEV, threshold=65)
    if name:
        name = fix_broken_devanagari_runs(name)
        if name.strip().lower() in ("नाव","name","नाम", "type"):
            name = None
    if address:
        address = fix_broken_devanagari_runs(address)
        if address.strip().lower() in ("address","type"):
            address = None

    # Normalize state: direct match first, then infer from district
    state = None
    for dev, sname in DEVANAGARI_TO_STATE.items():
        if dev in t:
            state = sname; break
    if not state:
        for k, v in STATES_CANONICAL.items():
            if k in t.lower():
                state = v; break
    if not state and district:
        # simple heuristic: if district matches known Pune, Mumbai etc.
        if any(d.lower() in district.lower() for d in ("pune","पुणे","mumbai","मुंबई","nagpur","नागपूर")):
            state = "Maharashtra"
        elif any(d.lower() in district.lower() for d in ("meerut","मेरठ","lucknow","लखनऊ")):
            state = "Uttar Pradesh"

    # Clean sections: remove impossible numbers, dedupe
    if sections:
        cleaned = []
        for s in sections:
            b = re.match(r"(\d{1,3})", s)
            if b:
                n = int(b.group(1))
                if 1 <= n <= SECTION_MAX:
                    cleaned.append(str(n))
        sections = dedupe_preserve_order(cleaned) if cleaned else None

    # acts normalization
    if acts:
        acts = [a for a in acts if a]
    # category mapping
    category = map_case_category(acts, sections, t)

    # oparty detection: prefer labelled presence of Accused/Complainant
    oparty = None
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", t, re.IGNORECASE):
        oparty = "Accused"
    elif re.search(r"\b(complainant|informant|तक्रारदार|सूचक)\b", t, re.IGNORECASE):
        oparty = "Complainant"
    else:
        # try to infer: if name extracted from 'accused' block? fallback to None
        oparty = None

    # jurisdiction inference
    jurisdiction = None; jurisdiction_type = None
    if re.search(r"\bPAN[_\s-]?INDIA\b|ALL[_\s-]?INDIA", t, re.IGNORECASE):
        jurisdiction = "PAN_INDIA"; jurisdiction_type = "PAN_INDIA"
    elif district:
        jurisdiction = district; jurisdiction_type = "DISTRICT"
    elif state:
        jurisdiction = state; jurisdiction_type = "STATE"

    out = {
        "year": year,
        "state_name": state,
        "dist_name": district,
        "police_station": police_station,
        "under_acts": acts,
        "under_sections": sections,
        "revised_case_category": category,
        "oparty": oparty,
        "name": name,
        "address": address,
        "jurisdiction": jurisdiction,
        "jurisdiction_type": jurisdiction_type
    }

    # Final safety: trim overly large strings, reject obvious placeholders
    for k, v in list(out.items()):
        if isinstance(v, str):
            val = v.strip()
            if val.lower() in ("n/a", "none", "null", "na", "name", "नाव", "type", ""):
                out[k] = None
            elif len(val) > 800:
                out[k] = val[:800] + "..."
    return out

# ------------------ Streamlit UI ------------------

st.title("FIR PII Extractor — Final (Aggressive, best-effort)")

st.write(
    "Upload FIR PDF(s) (scanned/digital) or paste FIR text. "
    "The extractor applies OCR cleanup, regex-first extraction, NER boost, and fuzzy repair to return clean PII."
)

with st.sidebar:
    st.header("Settings")
    use_ner = st.checkbox("Use NER fallback (HuggingFace) if available", value=NER_AVAILABLE)
    tesseract_langs = st.text_input("Tesseract languages (comma separated)", value="eng+hin+mar")
    st.caption("Install tesseract + hin/mar traineddata for best results on Devanagari PDF scans.")
    if NER_AVAILABLE:
        if st.button("Preload NER model (may take time)"):
            st.info("Loading NER model...")
            ner_pipe = load_ner_pipeline()
            if ner_pipe:
                st.success("NER loaded.")
            else:
                st.error("NER pipeline could not be loaded.")

uploaded = st.file_uploader("Upload FIR PDF(s)", type=["pdf"], accept_multiple_files=True)
pasted = st.text_area("Or paste FIR text here", height=300, placeholder="Paste FIR text from Elasticsearch or copy-paste here...")

ner_pipe = load_ner_pipeline() if (use_ner and NER_AVAILABLE) else None

if st.button("Extract PII"):
    results = {}
    # Process uploaded files
    if uploaded:
        for f in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            try:
                text = extract_text_from_pdf(tmp_path, tesseract_langs=tesseract_langs)
                if not text.strip():
                    st.warning(f"No text extracted from {f.name}.")
                pii = extract_pii_full(text, use_ner and (ner_pipe is not None), ner_pipe)
                results[f.name] = pii
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Process pasted text
    if pasted and pasted.strip():
        text = canonicalize_text(pasted)
        pii = extract_pii_full(text, use_ner and (ner_pipe is not None), ner_pipe)
        results["pasted_text"] = pii

    if not results:
        st.warning("Please upload one or more PDFs or paste FIR text.")
    else:
        st.subheader("Extracted PII (final)")
        st.json(results, expanded=True)
        out_json = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(out_json.encode()).decode()
        st.markdown(f"[Download JSON](data:application/json;base64,{b64})")
        st.success("Extraction finished. If fields are missing or noisy for some documents, share a sample PDF and I'll further tune normalization/fuzzy lists for that state's template.")
