# fir_pii_extractor_pro.py
"""
FIR PII Extractor — PRO (best-effort ChatGPT-style)
- PDF -> text (PyMuPDF/pdfplumber/OCR)
- Aggressive cleanup + Devanagari repair
- Multi-pattern regex for labeled fields
- NER fallback (optional)
- Fuzzy repair for district / police-station using rapidfuzz
- Candidate scoring & selection, filters placeholders and noisy OCR
"""

import streamlit as st
import fitz, pdfplumber, pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import process, fuzz

# optional transformers NER
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor — PRO", layout="wide")

# ------------------ SMALL CANONICAL DATA (extend as needed) ------------------
# Seed lists for fuzzy matching / repair. Add your full district list file to improve results.
STATE_CANON = {
    "maharashtra": "Maharashtra", "mumbai": "Maharashtra", "pune": "Maharashtra",
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "delhi": "Delhi", "karnataka": "Karnataka", "gujarat": "Gujarat",
    "bihar": "Bihar", "tamil nadu": "Tamil Nadu", "west bengal": "West Bengal",
}

DISTRICT_SEED = [
    "Pune", "Pune City", "Mumbai", "Mumbai City", "Nagpur", "Nashik",
    "Meerut", "Lucknow", "Varanasi", "Kanpur", "Noida", "Ghaziabad",
    "Ahmedabad", "Bengaluru", "Jaipur", "Patna"
]
DISTRICT_SEED_DEV = ["पुणे", "मुंबई", "नागपूर", "नाशिक", "मेरठ", "लखनऊ", "वाराणसी", "कानपुर"]

POLICE_PS_SEED = ["Bhosari", "Hadapsar", "Dadar", "Andheri", "Colaba", "Cyber Crime Cell"]
POLICE_PS_SEED_DEV = ["भोसरी", "हडपसर", "डादर", "अंधेरी"]

KNOWN_ACTS = {
    "ipc": "Indian Penal Code 1860",
    "indian penal code": "Indian Penal Code 1860",
    "indian penal code 1860": "Indian Penal Code 1860",
    "crpc": "Code of Criminal Procedure 1973",
    "arms act": "Arms Act 1959",
    "information technology": "Information Technology Act 2000",
    "it act": "Information Technology Act 2000",
    "ndps": "NDPS Act 1985",
    "pocso": "POCSO Act 2012",
    "maharashtra police": "Maharashtra Police Act 1951",
}

PLACEHOLDER_TOKENS = set([
    "name", "नाव", "नाम", "type", "address", "of p.s.", "then name of p.s.", "of p.s", "of ps"
])

SECTION_MAX = 999

# ------------------ Utilities: cleaning and repair ------------------

def remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def strip_nonessential_unicode(s: str) -> str:
    # Keep ASCII, Devanagari, punctuation; remove other weird symbols
    return re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—]", " ", s)

def collapse_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()

def fix_broken_devanagari_runs(s: str) -> str:
    # Collapse spaces between consecutive Devanagari characters repeatedly
    def once(x):
        return re.sub(r"([\u0900-\u097F])\s+([\u0900-\u097F])", r"\1\2", x)
    prev = None
    cur = s
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
    low = v.lower()
    # if it contains placeholders or too short noisy tokens
    if low in PLACEHOLDER_TOKENS:
        return None
    # if it is like ", then Name of P.S." reject
    if re.search(r"then\s+name|of\s+p\.?s\.?", low):
        return None
    # if it's just punctuation or single char
    if len(re.sub(r"\W+", "", v)) < 2:
        return None
    return v

# ------------------ Text extraction (PDF -> text) ------------------

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
    # Try text extraction back-to-back with cleaning
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

# ------------------ NER loader ------------------
@st.cache_resource
def load_ner():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Use indic-ner if available; fallback to multilingual models
        try:
            pipe = pipeline("token-classification", model="ai4bharat/indic-ner", aggregation_strategy="simple")
            return pipe
        except Exception:
            return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
    except Exception:
        return None

# ------------------ Candidate collection: regex-heavy ------------------

def find_year_candidates(text: str) -> List[str]:
    cands = []
    m = re.search(r"(?:Year|वर्ष|Date of FIR|Date)\s*[:\-]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m: cands.append(m.group(1))
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    if m2: cands.append(m2.group(0))
    return cands

def find_firno_candidates(text: str) -> List[str]:
    cands = []
    m = re.search(r"(?:FIR No|F\\.I\\.R\\.|FIR|F\\.I\\.R No|म. खब|F\\.No\\.|F No)\s*[:\-]?\s*([A-Za-z0-9\-\/]{3,60})", text, re.IGNORECASE)
    if m: cands.append(m.group(1).strip())
    return cands

def find_district_candidates(text: str) -> List[str]:
    cands = []
    patterns = [
        r"(?:District|Dist\.|जिला|जिल्हा|District Name)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9 ,./-]{2,80})",
        r"District\s*\([^)]+\)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            val = m.group(1).strip()
            if val: cands.append(val)
    return cands

def find_police_station_candidates(text: str) -> List[str]:
    cands = []
    patterns = [
        r"(?:Police Station|P\.S\.|PS|पोलीस ठाणे|पुलिस थाना|थाना)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})",
        r"(?:Station|Station Name)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9 .,-]{2,80})"
    ]
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            val = m.group(1).strip()
            if val: cands.append(val)
    # also try "Transferred to P.S. <name>"
    for m in re.finditer(r"Transferred to P\.S\.?\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{2,80})", text, re.IGNORECASE):
        cands.append(m.group(1).strip())
    return cands

def find_acts_candidates(text: str) -> List[str]:
    cands = []
    # scan for known act keywords and capture surrounding chunk
    for key in KNOWN_ACTS.keys():
        for m in re.finditer(re.escape(key), text, re.IGNORECASE):
            # capture a small window around match to help normalization
            start = max(0, m.start()-40); stop = min(len(text), m.end()+40)
            chunk = text[start:stop]
            cands.append(chunk)
    # fallback: any 'Act' mention lines
    for m in re.finditer(r"(?:Act|अधिनियम|कायदा)[^\n]{0,120}", text, re.IGNORECASE):
        cands.append(m.group(0))
    # normalize to canonical KNOWN_ACTS mapping
    normalized = []
    for c in cands:
        low = c.lower()
        picked = None
        for k, norm in KNOWN_ACTS.items():
            if k in low:
                picked = norm
                break
        if picked and picked not in normalized:
            normalized.append(picked)
    return normalized if normalized else []

def find_section_candidates(text: str) -> List[str]:
    secs = []
    # prefer sections near labels
    for m in re.finditer(r"(?:Section|Sections|U/s|U/s\.|धारा|कलम|Sect\.?)", text, re.IGNORECASE):
        window = text[m.start(): m.start()+300]
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                val = int(base.group(1))
                if 1 <= val <= SECTION_MAX:
                    secs.append(str(val))
    # fallback: global numeric occurrences but only keep >=10 to reduce noise
    if not secs:
        nums = re.findall(r"\b\d{1,3}\b", text)
        for n in nums:
            val = int(n)
            if val >= 10 and val <= SECTION_MAX:
                secs.append(str(val))
    # dedupe preserve order
    out = []
    for s in secs:
        if s not in out:
            out.append(s)
    return out

def find_name_candidates(text: str) -> List[str]:
    cands = []
    # many label variants
    patterns = [
        r"(?:Complainant|Informant|Complainant\/Informant|तक्रारदार|सूचक|प्राप्तकर्ता)[^\n]{0,80}Name\s*[:\-]?\s*([A-Za-z\u0900-\u097F .]{2,120})",
        r"(?:Name|नाव|नाम)\s*[:\-]?\s*([A-Za-z\u0900-\u097F .]{2,120})",
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})"  # fallback English capitalized name (riskier)
    ]
    for p in patterns:
        for m in re.finditer(p, text):
            v = m.group(1).strip()
            if v and len(v) > 2:
                cands.append(v)
    return dedupe_preserve_order(cands)

def find_address_candidates(text: str) -> List[str]:
    cands = []
    for m in re.finditer(r"(?:Address|पत्ता|पत्ता[:\-\)]*)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\- ]{6,250})", text, re.IGNORECASE):
        v = m.group(1).strip()
        if v:
            # cut at phone/mobile labels
            v = re.split(r"(?:Phone|Mobile|मोबाइल|मोबा|फोन|UID|Passport)", v, flags=re.IGNORECASE)[0].strip()
            cands.append(v)
    # fallback: sequences with pin codes
    for m in re.finditer(r"[A-Za-z\u0900-\u097F0-9, .\-]{10,200}\b\d{6}\b", text):
        cands.append(m.group(0).strip())
    return dedupe_preserve_order(cands)

# ------------------ NER booster ------------------

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

# ------------------ Fuzzy repair helpers ------------------

def fuzzy_repair_to_list(candidate: str, choices: List[str], threshold: int = 70) -> str:
    if not candidate or not choices:
        return candidate
    cand = fix_broken_devanagari_runs(candidate)
    best = process.extractOne(cand, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return candidate

# ------------------ Scoring and final selection ------------------

def choose_best(candidates: List[Tuple[str, str]]) -> Optional[str]:
    """
    candidates: list of tuples (value, source)
    source priority: labeled_regex > ner > fuzzy > fallback_regex
    We'll score based on source weight + length sanity.
    """
    if not candidates:
        return None
    weights = {"label": 100, "ner": 70, "fuzzy": 80, "fallback": 40, "act": 90}
    scored = []
    for val, src in candidates:
        if not val: continue
        v = filter_placeholder_candidate(val)
        if not v: continue
        score = weights.get(src, 50)
        # longer but reasonable names/addresses get small boost
        score += min(len(v), 40) / 4
        scored.append((score, v))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# ------------------ Final extraction orchestrator ------------------

def extract_all_fields(text: str, use_ner: bool = True, ner_pipe=None) -> Dict[str, Any]:
    t = canonicalize_text(text)

    # collect candidates from regex
    year_c = find_year_candidates(t)
    fir_c = find_firno_candidates(t)
    dist_c = find_district_candidates(t)
    ps_c = find_police_station_candidates(t)
    acts_c = find_acts_candidates(t)
    sections_c = find_section_candidates(t)
    name_c = find_name_candidates(t)
    addr_c = find_address_candidates(t)

    # ner fallback
    ner_candidates = {"PER": [], "LOC": [], "ORG": []}
    if use_ner and TRANSFORMERS_AVAILABLE and ner_pipe is not None:
        ner_candidates = ner_boost(t, ner_pipe)

    # assemble candidate lists with sources
    year_list = [(y, "label") for y in year_c]
    fir_list = [(f, "label") for f in fir_c]

    # district candidates: labelled regex first, then ner locs, then fuzzy repaired ones
    district_candidates = [(d, "label") for d in dist_c]
    for loc in ner_candidates.get("LOC", []):
        district_candidates.append((loc, "ner"))

    # police station candidates
    ps_candidates = [(p, "label") for p in ps_c]
    # name candidates
    name_candidates = [(n, "label") for n in name_c]
    for per in ner_candidates.get("PER", []):
        name_candidates.append((per, "ner"))

    # address candidates
    address_candidates = [(a, "label") for a in addr_c]
    for loc in ner_candidates.get("LOC", []):
        address_candidates.append((loc, "ner"))

    # acts
    acts_final = list(acts_c) if acts_c else None
    # sections: already a clean list - apply extra filter: keep only >=10 or seen near label
    sections_final = []
    if sections_c:
        for s in sections_c:
            try:
                n = int(re.match(r"(\d{1,3})", s).group(1))
                if n >= 10:
                    sections_final.append(str(n))
            except:
                pass
        sections_final = dedupe_preserve_order(sections_final) or None

    # fuzzy repair attempts for district/police:
    # repair district using seed lists
    district_candidates_repaired = []
    for val, src in district_candidates:
        repaired = fuzzy_repair_to_list(val, DISTRICT_SEED + DISTRICT_SEED_DEV, threshold=65)
        if repaired != val:
            district_candidates_repaired.append((repaired, "fuzzy"))
        else:
            district_candidates_repaired.append((val, src))

    # police station repair
    ps_candidates_repaired = []
    for val, src in ps_candidates:
        repaired = fuzzy_repair_to_list(val, POLICE_PS_SEED + POLICE_PS_SEED_DEV, threshold=65)
        if repaired != val:
            ps_candidates_repaired.append((repaired, "fuzzy"))
        else:
            ps_candidates_repaired.append((val, src))

    # now pick best for each field
    year_best = choose_best(year_list) or None
    dist_best = choose_best(district_candidates_repaired) or None
    ps_best = choose_best(ps_candidates_repaired) or None
    name_best = choose_best(name_candidates) or None
    addr_best = choose_best(address_candidates) or None

    # state detection: direct labeled, fallback by district fuzzy->state mapping, final: search known state words
    state_best = None
    # find words matching state canon
    for k, v in STATE_CANON.items():
        if k in t.lower() or v.lower() in t.lower():
            state_best = v; break
    if not state_best and dist_best:
        # heuristic mapping
        if any(d.lower() in dist_best.lower() for d in ("pune", "मुंबई", "pune city", "nagpur")):
            state_best = "Maharashtra"
        elif any(d.lower() in dist_best.lower() for d in ("meerut", "लखनऊ", "kanpur", "varanasi")):
            state_best = "Uttar Pradesh"

    # oparty detection
    oparty_best = None
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", t, re.IGNORECASE):
        oparty_best = "Accused"
    elif re.search(r"\b(complainant|informant|तक्रारदार|सूचक)\b", t, re.IGNORECASE):
        oparty_best = "Complainant"

    # case category mapping (basic)
    revised_cat = "OTHER"
    if acts_final:
        if any("Information Technology" in a or "IT Act" in a for a in acts_final):
            revised_cat = "CYBER_CRIME"
        elif any("Arms Act" in a for a in acts_final):
            revised_cat = "WEAPONS"
    if sections_final:
        sset = set(sections_final)
        if any(x in sset for x in ("354","376","509")):
            revised_cat = "SEXUAL_OFFENCE"
        elif "302" in sset:
            revised_cat = "MURDER"

    # jurisdiction inference
    jurisdiction = None; jurisdiction_type = None
    if dist_best:
        jurisdiction = dist_best; jurisdiction_type = "DISTRICT"
    elif state_best:
        jurisdiction = state_best; jurisdiction_type = "STATE"

    out = {
        "year": year_best,
        "state_name": state_best,
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

    # Final safety filters: trim huge blobs and reject placeholders
    for k, v in list(out.items()):
        if isinstance(v, str):
            if len(v) > 800:
                out[k] = v[:800] + "..."
            if v.strip().lower() in ("name", "नाव", "type", "address", "null", "none", ""):
                out[k] = None
    return out

# ------------------ Streamlit UI ------------------

st.title("FIR PII Extractor — PRO (Aggressive, best-effort)")

st.markdown(
    "- Upload FIR PDFs (scanned or digital) or paste FIR text.\n"
    "- The extractor applies cleaning, regex-first extraction, NER fallback and fuzzy repair."
)

with st.sidebar:
    st.header("Settings")
    use_ner = st.checkbox("Use NER fallback (if transformers available)", value=TRANSFORMERS_AVAILABLE)
    tesseract_langs = st.text_input("Tesseract languages", value="eng+hin+mar")
    st.markdown("Tip: install tesseract languages for best OCR (hin, mar).")

uploaded_files = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
pasted_text = st.text_area("Or paste FIR text here", height=300, placeholder="Paste FIR text here...")

ner_pipe = None
if use_ner and TRANSFORMERS_AVAILABLE:
    ner_pipe = load_ner()

if st.button("Extract PII"):
    results = {}
    # handle uploaded files
    if uploaded_files:
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read()); tmp_path = tmp.name
            try:
                txt = extract_text_from_pdf(tmp_path, tesseract_langs=tesseract_langs)
                if not txt.strip():
                    st.warning(f"No text extracted from {f.name}")
                res = extract_all_fields(txt, use_ner=use_ner, ner_pipe=ner_pipe)
                results[f.name] = res
            finally:
                try: os.remove(tmp_path)
                except: pass

    # handle pasted text
    if pasted_text and pasted_text.strip():
        txt = canonicalize_text(pasted_text)
        results["pasted_text"] = extract_all_fields(txt, use_ner=use_ner, ner_pipe=ner_pipe)

    if not results:
        st.warning("Please upload PDF(s) or paste FIR text.")
    else:
        st.subheader("Extracted PII (best-effort)")
        st.json(results, expanded=True)
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(payload.encode()).decode()
        st.markdown(f"[Download JSON](data:application/json;base64,{b64})")
        st.success("Extraction complete. If a specific FIR output remains noisy, upload that FIR and I'll tune fuzzy mappings/regexes for that template.")
