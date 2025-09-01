# fir_pii_extractor_ultimate.py
"""
FIR PII Extractor — ULTIMATE (final corrected)
- Single-file Streamlit app
- Robust PDF/text extraction (Hindi + English)
- Aggressive cleaning, Devanagari repair, regex-first candidates, optional NER, fuzzy repair, scoring
"""

import streamlit as st
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import process, fuzz

# Optional transformers NER
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor — ULTIMATE", layout="wide")

# ---------------- Constants (consistent names) ----------------
SECTION_MAX = 999
SECTION_MIN_KEEP = 10

PLACEHOLDERS = set([
    "name", "नाव", "नाम", "type", "address", "of p.s.", "then name of p.s.", "of p.s", "of ps"
])

STATE_CANON = {
    "maharashtra": "Maharashtra",
    "uttar pradesh": "Uttar Pradesh",
    "delhi": "Delhi",
    "karnataka": "Karnataka",
    "gujarat": "Gujarat",
    "bihar": "Bihar",
    "tamil nadu": "Tamil Nadu"
}

# Use consistent seed variable names (these are used by fuzzy repair)
DISTRICT_SEED = [
    "Pune", "Pune City", "Mumbai", "Mumbai City", "Nagpur", "Nashik",
    "Meerut", "Lucknow", "Varanasi", "Kanpur", "Noida", "Ghaziabad",
    "Ahmedabad", "Bengaluru", "Jaipur", "Patna"
]
DISTRICT_SEED_DEV = ["पुणे","मुंबई","नागपूर","नाशिक","मेरठ","लखनऊ","वाराणसी","कानपुर"]

POLICE_PS_SEED = ["Bhosari", "Hadapsar", "Dadar", "Andheri", "Colaba", "Cyber Crime Cell"]
POLICE_PS_SEED_DEV = ["भोसरी","हडपसर","डादर","अंधेरी"]

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

# ---------------- Utilities (all defined before use) ----------------

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for it in items:
        if it is None:
            continue
        s = it.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def strip_nonessential_unicode(s: str) -> str:
    return re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—]", " ", s)

def collapse_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()

def fix_broken_devanagari_runs(s: str) -> str:
    def once(x):
        return re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", x)
    prev = None; cur = s
    for _ in range(6):
        prev = cur
        cur = once(cur)
        if cur == prev:
            break
    return cur

def canonicalize_text(s: str) -> str:
    if not s:
        return ""
    s = remove_control_chars(s)
    s = strip_nonessential_unicode(s)
    s = collapse_spaces(s)
    s = fix_broken_devanagari_runs(s)
    return s

def filter_placeholder_candidate(c: Optional[str]) -> Optional[str]:
    if not c:
        return None
    v = c.strip()
    if not v:
        return None
    low = v.lower()
    if low in PLACEHOLDERS or re.search(r"\b(of p\.?s\.?|then name of p\.?s\.?)\b", low):
        return None
    if len(re.sub(r"\W+", "", v)) < 2:
        return None
    return v

# ---------------- Text extraction ----------------

def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        pages = [p.get_text("text") or "" for p in doc]
        return "\n".join(pages)
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

def extract_text_ocr(path: str, tesseract_langs: str) -> str:
    try:
        doc = fitz.open(path)
        out = []
        for p in doc:
            pix = p.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang=tesseract_langs))
        return "\n".join(out)
    except Exception:
        return ""

def extract_text_from_pdf(path: str, tesseract_langs: str = "eng+hin+mar") -> str:
    txt = extract_text_pymupdf(path)
    txt = canonicalize_text(txt)
    if len(txt) < 200:
        alt = extract_text_pdfplumber(path)
        alt = canonicalize_text(alt)
        if len(alt) > len(txt):
            txt = alt
    if len(txt) < 200:
        ocr = extract_text_ocr(path, tesseract_langs)
        ocr = canonicalize_text(ocr)
        if len(ocr) > len(txt):
            txt = ocr
    return canonicalize_text(txt)

# ---------------- NER loader ----------------

@st.cache_resource
def load_ner_pipe():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        try:
            return pipeline("token-classification", model="ai4bharat/indic-ner", aggregation_strategy="simple")
        except Exception:
            return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
    except Exception:
        return None

# ---------------- Candidate finders (regex) ----------------

def find_year_candidates(text: str) -> List[str]:
    out = []
    m = re.search(r"(?:Year|वर्ष|Date of FIR|Date)\s*[:\-]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m: out.append(m.group(1))
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    if m2: out.append(m2.group(0))
    return dedupe_preserve_order(out)

def find_district_candidates(text: str) -> List[str]:
    out = []
    patterns = [
        r"(?:District|Dist\.|जिला|जिल्हा|District Name)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9 ,./-]{2,80})",
        r"District\s*\([^)]+\)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

def find_police_station_candidates(text: str) -> List[str]:
    out = []
    patterns = [
        r"(?:Police Station|P\.S\.|PS\b|पोलीस ठाणे|पुलिस थाना|थाना)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})",
        r"(?:Station|Station Name)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            out.append(m.group(1).strip())
    for m in re.finditer(r"Transferred to P\.S\.?\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})", text, re.IGNORECASE):
        out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

def find_acts_candidates(text: str) -> List[str]:
    found = []
    for k in KNOWN_ACTS.keys():
        if re.search(re.escape(k), text, re.IGNORECASE):
            found.append(KNOWN_ACTS[k])
    for m in re.finditer(r"(?:Act|अधिनियम|कायदा)[^\n]{0,120}", text, re.IGNORECASE):
        chunk = m.group(0).lower()
        for k in KNOWN_ACTS.keys():
            if k in chunk:
                found.append(KNOWN_ACTS[k])
    return dedupe_preserve_order(found)

def find_section_candidates(text: str) -> List[str]:
    secs = []
    for m in re.finditer(r"(?:Section|Sections|U\/s|U\/s\.|धारा|कलम|Sect)\b", text, re.IGNORECASE):
        window = text[m.start(): m.start()+300]
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                try:
                    v = int(base.group(1))
                except Exception:
                    continue
                if 1 <= v <= SECTION_MAX:
                    secs.append(str(v))
    if not secs:
        nums = re.findall(r"\b\d{1,3}\b", text)
        for n in nums:
            try:
                v = int(n)
            except Exception:
                continue
            if v >= SECTION_MIN_KEEP and v <= SECTION_MAX:
                secs.append(str(v))
    return dedupe_preserve_order(secs)

def find_name_candidates(text: str) -> List[str]:
    out = []
    patterns = [
        r"(?:Complainant|Informant|तक्रारदार|सूचक|Complainant\/Informant)[^\n]{0,80}[:\-]?\s*([A-Za-z\u0900-\u097F .]{2,120})",
        r"(?:Name|नाव|नाम)\s*[:\-]?\s*([A-Za-z\u0900-\u097F .]{2,120})",
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            v = m.group(1).strip(); out.append(v)
    for m in re.finditer(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b", text):
        out.append(m.group(1).strip())
    return dedupe_preserve_order(out)

def find_address_candidates(text: str) -> List[str]:
    out = []
    for m in re.finditer(r"(?:Address|पत्ता|पत्ता[:\-\)]*)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\- ]{6,250})", text, re.IGNORECASE):
        v = m.group(1).strip()
        v = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Passport)", v, flags=re.IGNORECASE)[0].strip()
        out.append(v)
    for m in re.finditer(r"[A-Za-z\u0900-\u097F0-9, .\-]{10,200}\b\d{6}\b", text):
        out.append(m.group(0).strip())
    return dedupe_preserve_order(out)

# ---------------- NER booster ----------------

def ner_boost(text: str, ner_pipe) -> Dict[str, List[str]]:
    res = {"PER": [], "LOC": [], "ORG": []}
    if ner_pipe is None:
        return res
    try:
        ents = ner_pipe(text[:8000])
        for e in ents:
            grp = e.get("entity_group") or e.get("entity")
            w = e.get("word") or ""
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

# ---------------- Fuzzy repair ----------------

def fuzzy_repair_to_list(candidate: str, choices: List[str], threshold: int = 70) -> str:
    if not candidate or not choices:
        return candidate
    cand = fix_broken_devanagari_runs(candidate)
    best = process.extractOne(cand, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return candidate

# ---------------- Scoring ----------------

def choose_best(candidates: List[Tuple[str, str]]) -> Optional[str]:
    if not candidates:
        return None
    weights = {"label": 120, "ner": 80, "fuzzy": 100, "fallback": 40, "act": 110}
    scored = []
    for val, src in candidates:
        if not val:
            continue
        v = filter_placeholder_candidate(val)
        if not v:
            continue
        score = weights.get(src, 50)
        score += min(len(v), 60) / 6
        scored.append((score, v))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# ---------------- Orchestration ----------------

def extract_all_fields(text: str, use_ner: bool = True, ner_pipe=None) -> Dict[str, Any]:
    t = canonicalize_text(text)

    year_c = find_year_candidates(t)
    dist_c = find_district_candidates(t)
    ps_c = find_police_station_candidates(t)
    acts_c = find_acts_candidates(t)
    sections_c = find_section_candidates(t)
    name_c = find_name_candidates(t)
    addr_c = find_address_candidates(t)

    ner_c = {"PER": [], "LOC": [], "ORG": []}
    if use_ner and TRANSFORMERS_AVAILABLE and ner_pipe is not None:
        ner_c = ner_boost(t, ner_pipe)

    year_list = [(y, "label") for y in year_c]
    district_list = [(d, "label") for d in dist_c] + [(l, "ner") for l in ner_c.get("LOC", [])]
    ps_list = [(p, "label") for p in ps_c]
    name_list = [(n, "label") for n in name_c] + [(p, "ner") for p in ner_c.get("PER", [])]
    addr_list = [(a, "label") for a in addr_c] + [(l, "ner") for l in ner_c.get("LOC", [])]

    # fuzzy repair to canonical lists (consistent variable names)
    district_rep = []
    for val, src in district_list:
        rep = fuzzy_repair_to_list(val, DISTRICT_SEED + DISTRICT_SEED_DEV, threshold=65)
        if rep != val:
            district_rep.append((rep, "fuzzy"))
        else:
            district_rep.append((val, src))

    ps_rep = []
    for val, src in ps_list:
        rep = fuzzy_repair_to_list(val, POLICE_PS_SEED + POLICE_PS_SEED_DEV, threshold=65)
        if rep != val:
            ps_rep.append((rep, "fuzzy"))
        else:
            ps_rep.append((val, src))

    year_best = choose_best(year_list)
    dist_best = choose_best(district_rep)
    ps_best = choose_best(ps_rep)
    name_best = choose_best(name_list)
    addr_best = choose_best(addr_list)

    acts_final = acts_c if acts_c else None
    sections_final = sections_c if sections_c else None

    # state inference
    state_best = None
    for k, v in STATE_CANON.items():
        if k in t.lower() or v.lower() in t.lower():
            state_best = v; break
    if not state_best and dist_best:
        if any(x.lower() in dist_best.lower() for x in ["pune","mumbai","nagpur"]):
            state_best = "Maharashtra"
        elif any(x.lower() in dist_best.lower() for x in ["meerut","lucknow","kanpur"]):
            state_best = "Uttar Pradesh"

    oparty_best = None
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", t, re.IGNORECASE):
        oparty_best = "Accused"
    elif re.search(r"\b(complainant|informant|तक्रारदार|सूचक)\b", t, re.IGNORECASE):
        oparty_best = "Complainant"

    revised_cat = "OTHER"
    if acts_final and any("Information Technology" in a or "IT Act" in a for a in acts_final):
        revised_cat = "CYBER_CRIME"
    if sections_final:
        sset = set(sections_final)
        if any(x in sset for x in ("354","376","509")):
            revised_cat = "SEXUAL_OFFENCE"
        elif "302" in sset:
            revised_cat = "MURDER"

    jurisdiction = None; jurisdiction_type = None
    if dist_best:
        jurisdiction = dist_best; jurisdiction_type = "DISTRICT"
    elif state_best:
        jurisdiction = state_best; jurisdiction_type = "STATE"

    out = {
        "year": year_best,
        "state_name": filter_placeholder_candidate(state_best),
        "dist_name": filter_placeholder_candidate(dist_best),
        "police_station": filter_placeholder_candidate(ps_best),
        "under_acts": acts_final,
        "under_sections": sections_final,
        "revised_case_category": revised_cat,
        "oparty": oparty_best,
        "name": filter_placeholder_candidate(name_best),
        "address": filter_placeholder_candidate(addr_best),
        "jurisdiction": filter_placeholder_candidate(jurisdiction),
        "jurisdiction_type": jurisdiction_type
    }

    for k, v in list(out.items()):
        if isinstance(v, str):
            if len(v) > 1000:
                out[k] = v[:1000] + "..."
            if v.strip().lower() in ("name","नाव","type","address","null","none",""):
                out[k] = None
    return out

# ---------------- Streamlit UI ----------------

st.title("FIR PII Extractor — ULTIMATE")
st.write("Upload FIR PDFs (scanned/digital) or paste FIR text. Uses hybrid pipeline and fuzzy repair.")

with st.sidebar:
    st.header("Settings")
    use_ner = st.checkbox("Use NER fallback (if available)", value=TRANSFORMERS_AVAILABLE)
    tesseract_langs = st.text_input("Tesseract languages", value="eng+hin+mar")
    st.caption("Install tesseract language packs (hin, mar) on the machine for best OCR.")

uploaded = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
pasted = st.text_area("Or paste FIR text here", height=300)

ner_pipe = load_ner_pipe() if (use_ner and TRANSFORMERS_AVAILABLE) else None

if st.button("Extract PII"):
    results = {}
    # process uploaded PDFs
    if uploaded:
        for f in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
            try:
                txt = extract_text_from_pdf(tmp_path, tesseract_langs=tesseract_langs)
                if not txt.strip():
                    st.warning(f"No text extracted from {f.name}")
                results[f.name] = extract_all_fields(txt, use_ner=use_ner, ner_pipe=ner_pipe)
            finally:
                try: os.remove(tmp_path)
                except: pass

    # process pasted text
    if pasted and pasted.strip():
        txt = canonicalize_text(pasted)
        results["pasted_text"] = extract_all_fields(txt, use_ner=use_ner, ner_pipe=ner_pipe)

    if not results:
        st.warning("Please upload PDF(s) or paste FIR text.")
    else:
        st.subheader("Extracted PII (final)")
        st.json(results, expanded=True)
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(payload.encode()).decode()
        st.markdown(f"[Download JSON](data:application/json;base64,{b64})")
        st.success("Extraction complete.")
