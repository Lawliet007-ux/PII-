# fir_pii_extractor_improved.py
"""
FIR PII Extractor — IMPROVED (tuned for Marathi/Hindi + English FIRs)
- Single-file Streamlit app
- PDF -> text (PyMuPDF / pdfplumber / OCR fallback)
- Aggressive cleanup (Devanagari digit conversion, collapsing broken devanagari runs)
- Regex-first candidate collection in English/Hindi/Marathi
- Optional NER fallback (transformers) — used only as a backup
- Fuzzy repair for district/police station with seed lists; accepts CSV
- Debug mode shows intermediate candidate lists so you can tune quickly
"""

from __future__ import annotations
import streamlit as st
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import process, fuzz

# optional transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

st.set_page_config(page_title="FIR PII Extractor — IMPROVED", layout="wide")

# ------------------------------- Config / Seeds -------------------------------
SECTION_MAX = 999
SECTION_MIN_KEEP = 1    # we'll accept small sections when they're near "धारा/कलम/Section"
PLACEHOLDERS = set(["name", "नाव", "नाम", "type", "address", "of p.s.", "then name of p.s.", "of p.s", "of ps"])

# small canonical seed lists; you should expand these with full CSVs in production
DISTRICT_SEED = ["Pune", "Mumbai", "Nagpur", "Nashik", "Raigad", "Raigarh", "Meerut", "Lucknow", "Varanasi"]
DISTRICT_SEED_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","रायगड","मेरठ","लखनऊ","वाराणसी"]
POLICE_PS_SEED = ["Bhosari","Roha","Hadapsar","Andheri","Dadar"]
POLICE_PS_SEED_DEV = ["भोसरी","रोहा","हडपसर","अंधेरी","डादर"]

KNOWN_ACTS = {
    "ipc": "Indian Penal Code 1860",
    "indian penal code": "Indian Penal Code 1860",
    "crpc": "CrPC (Code of Criminal Procedure)",
    "arms act": "Arms Act 1959",
    "it act": "Information Technology Act 2000",
    "information technology": "Information Technology Act 2000",
    "bnss": "B.N.S.S. Act (as appearing in some FIRs)",
    "maharashtra police": "Maharashtra Police Act 1951",
}

# ------------------------------- Utilities -----------------------------------

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set(); out=[]
    for it in items:
        if it is None: continue
        s = it.strip()
        if not s: continue
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# convert Devanagari digits to ascii digits (०१२३४५६७८९ -> 0123456789)
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
def devanagari_to_ascii_digits(s: str) -> str:
    return s.translate(DEV_DIGITS)

def remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def strip_nonessential_unicode(s: str) -> str:
    # keep ASCII, Devanagari, punctuation; remove other odd symbols
    return re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—]", " ", s)

def collapse_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()

def fix_broken_devanagari_runs(s: str) -> str:
    # collapse spaces between Devanagari letters iteratively
    def once(x): return re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", x)
    prev=None; cur=s
    for _ in range(6):
        prev=cur; cur=once(cur)
        if cur==prev: break
    return cur

def canonicalize_text(s: str) -> str:
    if not s: return ""
    s = devanagari_to_ascii_digits(s)
    s = remove_control_chars(s)
    s = strip_nonessential_unicode(s)
    s = collapse_spaces(s)
    s = fix_broken_devanagari_runs(s)
    return s

def filter_placeholder_candidate(c: Optional[str]) -> Optional[str]:
    if not c: return None
    v = c.strip()
    if not v: return None
    low = v.lower()
    if low in PLACEHOLDERS or re.search(r"\b(of p\.?s\.?|then name of p\.?s\.?)\b", low):
        return None
    if len(re.sub(r"\W+","", v)) < 2:
        return None
    return v

# ------------------------------- PDF -> Text ---------------------------------

def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        pages = [p.get_text("text") or "" for p in doc]
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_pdfplumber(path: str) -> str:
    try:
        out=[]
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                out.append(p.extract_text() or "")
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_ocr(path: str, tesseract_langs: str="eng+hin+mar") -> str:
    try:
        doc = fitz.open(path)
        out=[]
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB",[pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang=tesseract_langs))
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf(path: str, tesseract_langs: str="eng+hin+mar") -> str:
    txt = extract_text_pymupdf(path)
    txt = canonicalize_text(txt)
    if len(txt) < 200:
        alt = extract_text_pdfplumber(path); alt = canonicalize_text(alt)
        if len(alt) > len(txt): txt = alt
    if len(txt) < 200:
        ocr = extract_text_ocr(path, tesseract_langs); ocr = canonicalize_text(ocr)
        if len(ocr) > len(txt): txt = ocr
    return canonicalize_text(txt)

# ------------------------------- Candidate Finders ---------------------------

# YEAR
def find_year_candidates(text: str) -> List[str]:
    out=[]
    m = re.search(r"(?:Year|वर्ष|Year\s*\(|वर्ष\s*\(|Year\s*[:\-])\s*([12][0-9]{3})", text, re.IGNORECASE)
    if m: out.append(m.group(1))
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    if m2: out.append(m2.group(0))
    return dedupe_preserve_order(out)

# POLICE STATION
def find_police_station_candidates(text: str) -> List[str]:
    out=[]
    patterns = [
        r"(?:P\.S\.|P\.S|Police Station|पोलीस ठाणे|पुलिस थाना|थाना)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})",
        r"(?:Station)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip()
            if v: out.append(v)
    # also "Name of P.S." patterns maybe after "then"
    for m in re.finditer(r"Name of P\.S\.?\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})", text, re.IGNORECASE):
        out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

# DISTRICT
def find_district_candidates(text: str) -> List[str]:
    out=[]
    for m in re.finditer(r"(?:District|District\s*\(|जिला|जिल्हा|District Name)\s*[:\-\)]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})", text, re.IGNORECASE):
        out.append(m.group(1).strip())
    # trailing "District (State)" lines common in FIRs
    for m in re.finditer(r"District\s*\(?([A-Za-z\u0900-\u097F ]{2,80})\)?\s*\(.*\)", text, re.IGNORECASE):
        out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

# ACTS
def find_acts_candidates(text: str) -> List[str]:
    found=[]
    low = text.lower()
    for k,v in KNOWN_ACTS.items():
        if k in low:
            if v not in found: found.append(v)
    # look for lines containing 'Act' or 'अधिनियम' and capture chunk
    for m in re.finditer(r"(?:Act|अधिनियम|कायदा|अनिनयम)[^\n]{0,120}", text, re.IGNORECASE):
        chunk = m.group(0)
        for k,v in KNOWN_ACTS.items():
            if k in chunk.lower() and v not in found:
                found.append(v)
    return dedupe_preserve_order(found)

# SECTIONS
def find_section_candidates(text: str) -> List[str]:
    secs=[]
    # look near 'धारा', 'कलम', 'Section', 'U/s'
    for m in re.finditer(r"(?:धारा|कलम|Section|Sect\.|U/s|U/s\.)", text, re.IGNORECASE):
        window = text[m.start(): m.start()+250]
        # capture patterns like 37(1), 37(1)(a), 154, 3(25), 285 etc.
        nums = re.findall(r"\b\d{1,3}(?:\([0-9A-Za-z]+\))*\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                try:
                    val = int(base.group(1))
                except:
                    continue
                if 1 <= val <= SECTION_MAX:
                    secs.append(str(val))
    # fallback: if none found via labels, pick reasonable numbers >= SECTION_MIN_KEEP
    if not secs:
        for n in re.findall(r"\b\d{1,3}\b", text):
            try:
                v = int(n)
            except:
                continue
            if v >= SECTION_MIN_KEEP and v <= SECTION_MAX:
                secs.append(str(v))
    return dedupe_preserve_order(secs)

# NAMES
def find_name_candidates(text: str) -> List[str]:
    out=[]
    # labelled forms (complainant/informant/accused)
    patterns = [
        r"(?:Complainant|Informant|Complainant\/Informant|तक्रारदार|सूचक|अर्जदार|शिकायतदार)[^\n]{0,120}[:\-\)]?\s*([A-Za-z\u0900-\u097F .]{2,160})",
        r"(?:Name|नाव|नाम)\s*[:\-\)]\s*([A-Za-z\u0900-\u097F .]{2,160})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip()
            out.append(v)
    # accused block (look for "Details of accused")
    for m in re.finditer(r"(?:Details of known|Details of known / suspected|आरोप|आरोपी)[^\n]{0,160}", text, re.IGNORECASE):
        # capture some lines after label
        start = m.start()
        window = text[start:start+300]
        # try to extract probable names: sequences with Devanagari or capitalized latin
        names = re.findall(r"([A-Za-z\u0900-\u097F]{2,40}(?:\s+[A-Za-z\u0900-\u097F]{2,40}){0,3})", window)
        for nm in names:
            out.append(nm.strip())
    # fallback: some capitalized english name sequences
    for m in re.finditer(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b", text):
        out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

# ADDRESS
def find_address_candidates(text: str) -> List[str]:
    out=[]
    for m in re.finditer(r"(?:Address|पत्ता|Address\s*\(|पत्ता\s*\(|Address[:\-\)])\s*[:\-\)]?\s*([A-Za-z0-9\u0900-\u097F,./\-\n ]{8,300})", text, re.IGNORECASE):
        v = m.group(1).strip()
        v = re.split(r"(?:Mobile|मोबाइल|Phone|फोन|UID|Aadhar|पॅन|PAN|Passport)", v, flags=re.IGNORECASE)[0].strip()
        out.append(" ".join(v.split()))
    # fallback: lines containing district + possible pincode
    for m in re.finditer(r"([A-Za-z\u0900-\u097F0-9,.\- ]{8,200}\b\d{6}\b)", text):
        out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

# ------------------------------- NER (optional) -------------------------------

def load_ner_pipe():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # try indic-ner then fallback multilingual
        try:
            return pipeline("token-classification", model="ai4bharat/indic-ner", aggregation_strategy="simple")
        except Exception:
            return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
    except Exception:
        return None

def ner_boost(text: str, ner_pipe) -> Dict[str, List[str]]:
    res={"PER":[], "LOC":[], "ORG":[]}
    if ner_pipe is None:
        return res
    try:
        ents = ner_pipe(text[:8000])
        for e in ents:
            grp = e.get("entity_group") or e.get("entity")
            w = e.get("word") or ""
            w = w.strip()
            if not w: continue
            if grp in ("PER","PERSON"): res["PER"].append(w)
            elif grp in ("LOC","LOCATION","GPE"): res["LOC"].append(w)
            elif grp in ("ORG",): res["ORG"].append(w)
    except Exception:
        pass
    for k in res: res[k] = dedupe_preserve_order(res[k])
    return res

# ------------------------------- Fuzzy repair --------------------------------

def fuzzy_repair_to_list(candidate: str, choices: List[str], threshold: int = 68) -> str:
    if not candidate or not choices:
        return candidate
    cand = fix_broken_devanagari_runs(candidate)
    best = process.extractOne(cand, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return candidate

# ------------------------------- Scoring / Choose -----------------------------

def choose_best(candidates: List[Tuple[str,str]]) -> Optional[str]:
    """
    candidates: list of (value, source) where source in {'label','ner','fuzzy','fallback'}
    weight priority: label > fuzzy > ner > fallback
    """
    if not candidates: return None
    weights = {"label":120, "fuzzy":100, "ner":80, "fallback":40, "act":110}
    scored=[]
    for val, src in candidates:
        if not val: continue
        v = filter_placeholder_candidate(val)
        if not v: continue
        score = weights.get(src, 50)
        # penalize if it's a leftover like "then" or overly long paragraphs
        if len(v) > 500: score -= 40
        # small boost for moderate length
        score += min(len(v), 80)/8
        scored.append((score,v))
    if not scored: return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# ------------------------------- Orchestration -------------------------------

def extract_all_fields(text: str, use_ner: bool=True, ner_pipe=None, debug: bool=False) -> Dict[str,Any]:
    t = canonicalize_text(text)

    # Gather candidates via regex
    year_c = find_year_candidates(t)
    ps_c = find_police_station_candidates(t)
    dist_c = find_district_candidates(t)
    acts_c = find_acts_candidates(t)
    sections_c = find_section_candidates(t)
    name_c = find_name_candidates(t)
    addr_c = find_address_candidates(t)

    # NER fallback
    ner_c = {"PER":[],"LOC":[],"ORG":[]}
    if use_ner and TRANSFORMERS_AVAILABLE and ner_pipe:
        ner_c = ner_boost(t, ner_pipe)

    # Prepare candidate lists with source tags
    year_list = [(y,"label") for y in year_c]

    ps_list = [(p,"label") for p in ps_c] + [(p,"ner") for p in ner_c.get("LOC",[])]
    dist_list = [(d,"label") for d in dist_c] + [(l,"ner") for l in ner_c.get("LOC",[])]
    name_list = [(n,"label") for n in name_c] + [(p,"ner") for p in ner_c.get("PER",[])]
    addr_list = [(a,"label") for a in addr_c] + [(l,"ner") for l in ner_c.get("LOC",[])]

    # Fuzzy repair district and police station to seeds
    dist_rep=[]
    for val, src in dist_list:
        rep = fuzzy_repair_to_list(val, DISTRICT_SEED + DISTRICT_SEED_DEV, threshold=65)
        if rep!=val:
            dist_rep.append((rep,"fuzzy"))
        else:
            dist_rep.append((val,src))

    ps_rep=[]
    for val, src in ps_list:
        rep = fuzzy_repair_to_list(val, POLICE_PS_SEED + POLICE_PS_SEED_DEV, threshold=65)
        if rep!=val:
            ps_rep.append((rep,"fuzzy"))
        else:
            ps_rep.append((val,src))

    # Choose best
    year_best = choose_best(year_list)
    ps_best = choose_best(ps_rep)
    dist_best = choose_best(dist_rep)
    name_best = choose_best(name_list)
    addr_best = choose_best(addr_list)

    # Normalize acts and sections
    under_acts = acts_c if acts_c else None
    under_sections = sections_c if sections_c else None

    # Revised category mapping
    revised_case_category = "OTHER"
    if under_acts:
        if any("Information Technology" in a or "IT Act" in a for a in under_acts):
            revised_case_category = "CYBER_CRIME"
        elif any("Arms" in a for a in under_acts):
            revised_case_category = "WEAPONS"
    if under_sections:
        sset = set(under_sections)
        if any(x in sset for x in ("354","376","509")):
            revised_case_category = "SEXUAL_OFFENCE"
        if "302" in sset:
            revised_case_category = "MURDER"

    # Oparty detection heuristics
    oparty = None
    if re.search(r"\b(आरोपी|accused|प्रतिवादी)\b", t, re.IGNORECASE):
        oparty = "Accused"
    elif re.search(r"\b(तक्रारदार|complainant|informant|सूचक)\b", t, re.IGNORECASE):
        oparty = "Complainant"

    # Jurisdiction inference
    jurisdiction = None; jurisdiction_type=None
    if dist_best:
        jurisdiction = dist_best; jurisdiction_type="DISTRICT"
    else:
        # try to infer state by seed matching
        for s in DISTRICT_SEED:
            if s.lower() in t.lower():
                jurisdiction = s; jurisdiction_type="DISTRICT"; break

    # Sanitize name/address/dist fields
    name_best = filter_placeholder_candidate(name_best)
    addr_best = filter_placeholder_candidate(addr_best)
    dist_best = filter_placeholder_candidate(dist_best)
    ps_best = filter_placeholder_candidate(ps_best)

    result = {
        "year": year_best,
        "state_name": None,          # optional: can infer from district if you expand mapping
        "dist_name": dist_best,
        "police_station": ps_best,
        "under_acts": under_acts,
        "under_sections": under_sections,
        "revised_case_category": revised_case_category,
        "oparty": oparty,
        "name": name_best,
        "address": addr_best,
        "jurisdiction": jurisdiction,
        "jurisdiction_type": jurisdiction_type
    }

    if debug:
        debug_info = {
            "raw_text_snippet": t[:2000],
            "year_candidates": year_c,
            "ps_candidates": ps_c,
            "dist_candidates": dist_c,
            "acts_candidates": acts_c,
            "section_candidates": sections_c,
            "name_candidates": name_c,
            "address_candidates": addr_c,
            "ner_candidates": ner_c
        }
        result["_debug"] = debug_info

    return result

# ------------------------------- Streamlit UI -------------------------------

st.title("FIR PII Extractor — IMPROVED (Marathi/Hindi + English)")
st.write("Upload PDFs (scanned/digital) or paste text. Turn on Debug to see intermediate candidates and tune.")

with st.sidebar:
    st.header("Settings")
    use_ner = st.checkbox("Use NER fallback (transformers) — optional", value=False)
    tesseract_langs = st.text_input("Tesseract languages", value="eng+hin+mar")
    debug = st.checkbox("Debug mode (show candidate lists)", value=True)
    st.markdown("Tip: in Debug mode the extractor shows intermediate candidate lists (helpful for tuning).")

uploaded_files = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
pasted_text = st.text_area("Or paste FIR text here (raw OCR / copy-paste)", height=300)

ner_pipe = None
if use_ner and TRANSFORMERS_AVAILABLE:
    ner_pipe = load_ner_pipe()

if st.button("Extract PII"):
    results = {}
    # handle uploaded
    if uploaded_files:
        for up in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.read()); tmp_path = tmp.name
            try:
                txt = extract_text_from_pdf(tmp_path, tesseract_langs=tesseract_langs)
                if not txt.strip():
                    st.warning(f"No text extracted from {up.name}")
                res = extract_all_fields(txt, use_ner=use_ner, ner_pipe=ner_pipe, debug=debug)
                results[up.name] = res
            finally:
                try: os.remove(tmp_path)
                except: pass

    # handle pasted
    if pasted_text and pasted_text.strip():
        txt = canonicalize_text(pasted_text)
        res = extract_all_fields(txt, use_ner=use_ner, ner_pipe=ner_pipe, debug=debug)
        results["pasted_text"] = res

    if not results:
        st.warning("Please upload at least one PDF or paste text.")
    else:
        st.subheader("Extracted PII")
        st.json(results, expanded=True)
        out = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(out.encode()).decode()
        st.markdown(f"[Download JSON](data:application/json;base64,{b64})")
        st.success("Done. Inspect _debug (if enabled) to tune regex/seed lists.")

# ------------------------------- End file -----------------------------------
