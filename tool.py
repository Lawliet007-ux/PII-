"""
fir_pii_extractor.py
Hybrid FIR PII extractor (PDF + pasted text)
- Text extraction: PyMuPDF -> pdfplumber -> OCR (pytesseract) fallback
- Extraction: targeted regex (strict) -> NER (optional) -> normalization & validation
- Outputs clean JSON for requested fields:
  year, state_name, dist_name, police_station, under_acts, under_sections,
  revised_case_category, oparty, name, address, jurisdiction, jurisdiction_type
"""

import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io, os, re, json, unicodedata, tempfile, base64
from typing import List, Dict, Any, Optional

# Optional transformers NER
NER_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    NER_AVAILABLE = False

st.set_page_config(page_title="FIR PII Extractor — Hybrid", layout="wide")

# ---------------- Constants & dictionaries ----------------

# Known states (English lowercase -> normalized)
INDIAN_STATES = {
    "maharashtra": "Maharashtra",
    "mumbai": "Maharashtra",
    "delhi": "Delhi",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "karnataka": "Karnataka",
    "punjab": "Punjab",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "bihar": "Bihar",
    "west bengal": "West Bengal",
    "tamil nadu": "Tamil Nadu",
    "madhya pradesh": "Madhya Pradesh",
    "jharkhand": "Jharkhand",
    "odisha": "Odisha",
    "telangana": "Telangana",
    "kerala": "Kerala",
    "haryana": "Haryana",
    # add more as required
}

# Devanagari -> normalized state
STATE_DEVANAGARI_MAP = {
    "महाराष्ट्र": "Maharashtra",
    "मुम्बई": "Maharashtra",
    "म्हाराष्ट्र": "Maharashtra",
    "दिल्ली": "Delhi",
    "उत्तर प्रदेश": "Uttar Pradesh",
    "कर्नाटक": "Karnataka",
    "पुणे": "Maharashtra",
    "पुणे शहर": "Maharashtra",
    # add as you encounter more variants
}

KNOWN_ACTS = [
    "Indian Penal Code", "IPC", "IPC 1860", "Indian Penal Code 1860",
    "Code of Criminal Procedure", "CrPC", "CrPC 1973",
    "Arms Act", "Arms Act 1959", "Arms Act, 1959",
    "Maharashtra Police Act", "Maharashtra Police Act 1951",
    "Information Technology Act", "IT Act", "IT Act 2000",
    "NDPS Act", "NDPS Act 1985", "POCSO", "POCSO Act 2012"
]

SECTION_MAX = 600

CATEGORY_RULES = [
    ("WEAPONS", {"acts": ["Arms Act"], "sections": ["25", "3"]}),
    ("SEXUAL_OFFENCE", {"sections": ["354", "376", "509", "354A", "354B", "354C", "354D"]}),
    ("THEFT", {"sections": ["379", "380", "454", "457"]}),
    ("ROBBERY", {"sections": ["392", "395", "397"]}),
    ("HURT_ASSAULT", {"sections": ["323", "325", "326", "341", "342", "504", "506"]}),
    ("NARCOTICS", {"acts": ["NDPS"]}),
    ("CYBER_IT", {"acts": ["IT Act", "Information Technology Act"]}),
    ("PUBLIC_ORDER", {"sections": ["135", "37"], "acts": ["Maharashtra Police Act"]}),
]

# ---------------- Utilities ----------------

def clean_ocr_text(text: str) -> str:
    """
    Normalize unicode, strip control characters and aggressive OCR garbage,
    collapse whitespace.
    """
    if not text:
        return ""
    # remove control chars (category 'C')
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # replace weird punctuation sequences with space
    text = re.sub(r"[^\S\r\n]+", " ", text)  # collapse non-newline whitespace
    # remove isolated weird unicode ranges often produced by bad OCR
    text = re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\u0964\u0965\u0020-\u007E\n,:()\-/.]", " ", text)
    # collapse whitespace and trim
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return text

def safe_first(lst: List[Any]) -> Optional[Any]:
    return lst[0] if lst else None

def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for i in items:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

# ---------------- Text extraction ----------------

def extract_text_pymupdf(path: str) -> str:
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception:
        return ""
    return text

def extract_text_pdfplumber(path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text() or ""
                text += t + "\n"
    except Exception:
        return ""
    return text

def extract_text_ocr(path: str, tesseract_langs: str = "eng+hin+mar") -> str:
    """
    Render pages with PyMuPDF and OCR using pytesseract. Languages can be adjusted.
    """
    text = ""
    try:
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            t = pytesseract.image_to_string(img, lang=tesseract_langs)
            text += t + "\n"
    except Exception:
        return ""
    return text

def extract_text_from_pdf(path: str) -> str:
    """
    Try PyMuPDF -> pdfplumber -> OCR falling back as needed.
    """
    txt = extract_text_pymupdf(path)
    if len(clean_ocr_text(txt)) < 200:
        alt = extract_text_pdfplumber(path)
        if len(clean_ocr_text(alt)) > len(clean_ocr_text(txt)):
            txt = alt
    if len(clean_ocr_text(txt)) < 200:
        ocr = extract_text_ocr(path)
        if len(clean_ocr_text(ocr)) > len(clean_ocr_text(txt)):
            txt = ocr
    return clean_ocr_text(txt)

# ---------------- NER loading & wrapper ----------------

@st.cache_resource
def load_ner_pipeline():
    if not NER_AVAILABLE:
        return None
    # Try to load a lightweight multilingual NER. If offline or fails, return None.
    try:
        # By default try a public multilingual NER; replace with fine-tuned model if available.
        return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
    except Exception:
        try:
            return pipeline("token-classification", model="xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")
        except Exception:
            return None

def ner_extract_candidates(text: str, ner_pipe) -> Dict[str, List[str]]:
    """
    Return tokens grouped by category: PER, LOC, ORG
    """
    cands = {"PER": [], "LOC": [], "ORG": []}
    if ner_pipe is None:
        return cands
    try:
        ents = ner_pipe(text[:8000])
        for e in ents:
            grp = e.get("entity_group") or e.get("entity")
            word = e.get("word") or e.get("word")
            if not word:
                continue
            w = word.strip()
            if not w:
                continue
            if grp in ("PER", "PERSON"):
                cands["PER"].append(w)
            elif grp in ("LOC", "LOCATION", "GPE"):
                cands["LOC"].append(w)
            elif grp in ("ORG",):
                cands["ORG"].append(w)
    except Exception:
        pass
    # dedupe
    for k in list(cands.keys()):
        cands[k] = dedupe_preserve_order(cands[k])
    return cands

# ---------------- Field extraction helpers ----------------

def extract_year(text: str) -> Optional[str]:
    # prefer labeled 'Year' in Hindi/English then general 4-digit
    m = re.search(r"(?:Year|वर्ष|वष|Date\s*[:\-]\s*)\s*[:\-]?\s*((?:19|20)\d{2})", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m2 = re.search(r"\b(19|20)\d{2}\b", text)
    return m2.group(0) if m2 else None

def normalize_state(text: str) -> Optional[str]:
    # search Devanagari first, then English aliases
    for dev, norm in STATE_DEVANAGARI_MAP.items():
        if dev in text:
            return norm
    low = text.lower()
    for key, norm in INDIAN_STATES.items():
        if key in low:
            return norm
    # labeled capture fallback
    m = re.search(r"(?:State|राज्य)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{3,60})", text, re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        # try again
        for dev, norm in STATE_DEVANAGARI_MAP.items():
            if dev in cand:
                return norm
        for key, norm in INDIAN_STATES.items():
            if key in cand.lower():
                return norm
    return None

def extract_district(text: str) -> Optional[str]:
    m = re.search(r"(?:District|District[:\s]|जिला|जिल्हा)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9,/-]{3,80})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # look for patterns like 'District (....): <name>'
    m2 = re.search(r"District\s*\([^)]+\)\s*[:\-]?\s*([A-Za-z\u0900-\u097F ]{3,80})", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def extract_police_station(text: str) -> Optional[str]:
    # strict label-based extraction stopping at newline or punctuation
    m = re.search(r"(?:Police\s*Station|P\.?\s*S\.?|पोलीस\s*ठाणे|पुलिस\s*थाना|ठाणे)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9\-/,.]{2,80})", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # patterns like "PS <name>"
    m2 = re.search(r"\bPS\s+([A-Za-z\u0900-\u097F\- ]{2,60})", text)
    if m2:
        return m2.group(1).strip()
    return None

def extract_acts(text: str) -> Optional[List[str]]:
    found = []
    # capture near 'Acts' label first
    for m in re.finditer(r"(?:Acts|Act|अधिनियम|कायदा)\s*[:\-]?\s*([^\n]{0,150})", text, re.IGNORECASE):
        chunk = m.group(1)
        for a in KNOWN_ACTS:
            if re.search(re.escape(a), chunk, re.IGNORECASE):
                found.append(a)
    # global scan fallback
    for a in KNOWN_ACTS:
        if re.search(re.escape(a), text, re.IGNORECASE):
            found.append(a)
    if not found:
        return None
    # normalize common variants
    norm = []
    for a in dedupe_preserve_order(found):
        if re.search(r"भारतीय दंड संहिता|Indian Penal Code|IPC", a, re.IGNORECASE):
            norm.append("Indian Penal Code 1860")
        elif re.search(r"Arms Act|शस्त्र", a, re.IGNORECASE):
            norm.append("Arms Act 1959")
        elif re.search(r"Maharashtra Police", a, re.IGNORECASE):
            norm.append("Maharashtra Police Act 1951")
        elif re.search(r"CrP|Cr.P.C|Code of Criminal Procedure|दण्ड प्रक्रिया", a, re.IGNORECASE):
            norm.append("Code of Criminal Procedure 1973")
        elif re.search(r"Information Technology|IT Act|सूचना प्रौद्योगिकी", a, re.IGNORECASE):
            norm.append("Information Technology Act 2000")
        elif re.search(r"NDPS", a, re.IGNORECASE):
            norm.append("NDPS Act 1985")
        else:
            norm.append(a)
    return dedupe_preserve_order(norm)

def extract_sections(text: str) -> Optional[List[str]]:
    # look in vicinity of 'Section' labels
    sections = []
    for m in re.finditer(r"(?:Section|Sections|U/s|U/s\.|dhara|धारा|कलम|Sect)\b", text, re.IGNORECASE):
        window = text[m.start(): m.start()+300]
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", window)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                v = int(base.group(1))
                if 1 <= v <= SECTION_MAX:
                    sections.append(n)
    # fallback: any section-like numbers in document
    if not sections:
        nums = re.findall(r"\b\d{1,3}[A-Z]?(?:\([0-9A-Za-z]+\))?\b", text)
        for n in nums:
            base = re.match(r"(\d{1,3})", n)
            if base:
                v = int(base.group(1))
                if 1 <= v <= SECTION_MAX:
                    sections.append(n)
    sections = dedupe_preserve_order(sections)
    return sections if sections else None

def extract_name_by_label(text: str) -> Optional[str]:
    # look for labelled 'Name' fields in various languages, stop at newline or comma
    m = re.search(r"(?:Name|नाव|नाव[:\-\) ]+)\s*[:\-]?\s*([A-Z\u0900-\u097F][A-Za-z\u0900-\u097F .]{2,120})", text)
    if m:
        cand = m.group(1).strip()
        # trim if it's extremely long (it's probably a paragraph)
        cand = cand.split("\n")[0].strip()
        if len(cand) <= 120:
            return cand
    # look for 'Complainant' labelled name
    m2 = re.search(r"(?:Complainant|Informant|तक्रारदार|सूचक)[^\n]{0,50}(?:Name|नाव)?\s*[:\-]?\s*([A-Z\u0900-\u097F][A-Za-z\u0900-\u097F .]{2,120})", text, re.IGNORECASE)
    if m2:
        return m2.group(1).split("\n")[0].strip()
    return None

def extract_address_by_label(text: str) -> Optional[str]:
    # match "Address" or "पत्ता", followed by colon/dash, then capture up to 200 chars
    m = re.search(
        r"(?:Address|पत्ता)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\- ]{4,200})",
        text,
        re.IGNORECASE
    )
    if m:
        addr = m.group(1).strip()
        # stop at phone/mobile/UID labels if they appear
        addr = re.split(r"(?:Phone|Mobile|मोबा|फोन|UID|Passport)", addr)[0].strip()
        return addr
    return None


# ---------------- Master extraction ----------------

def categorize_case(acts: Optional[List[str]], sections: Optional[List[str]], text: str) -> str:
    if not acts and not sections:
        return "OTHER"
    acts_text = " | ".join(acts) if acts else ""
    sections_upper = [s.upper() for s in sections] if sections else []
    for label, rules in CATEGORY_RULES:
        hit = False
        if "acts" in rules and acts:
            for a in rules["acts"]:
                if a.lower() in acts_text.lower():
                    hit = True; break
        if not hit and "sections" in rules and sections:
            for sec in rules["sections"]:
                # exact or startswith for e.g. 354A vs 354
                if any(s.upper().startswith(sec.upper()) for s in sections_upper):
                    hit = True; break
        if not hit and "keywords" in rules:
            for kw in rules["keywords"]:
                if re.search(kw, text, re.IGNORECASE):
                    hit = True; break
        if hit:
            return label
    return "OTHER"

def extract_pii_from_text(text: str, use_ner: bool = True, ner_pipe=None) -> Dict[str, Any]:
    """
    Hybrid extraction: regex-first, then NER fallback, then normalization/validation.
    Returns the canonical JSON structure.
    """
    t = clean_ocr_text(text)

    # core fields via regex/heuristics
    year = extract_year(t)
    state = normalize_state(t)
    district = extract_district(t)
    police_station = extract_police_station(t)
    acts = extract_acts(t)
    sections = extract_sections(t)

    # minimal validation / cleaning
    if sections:
        # remove obviously bogus like '00', '000' etc
        sections = [s for s in sections if re.match(r"\d{1,3}", s) and 1 <= int(re.match(r"(\d{1,3})", s).group(1)) <= SECTION_MAX]
        sections = dedupe_preserve_order(sections)
        if not sections:
            sections = None

    # use NER only as fallback / boost for name/address/location
    per = []; locs = []; orgs = []
    if use_ner and ner_pipe:
        ents = ner_extract_candidates(t, ner_pipe)
        per, locs, orgs = ents.get("PER", []), ents.get("LOC", []), ents.get("ORG", [])

    # Name extraction: prefer labeled name -> NER -> heuristics
    name = extract_name_by_label(t)
    if not name and per:
        # choose the longest PER candidate
        name = max(per, key=len) if per else None
    if name:
        # normalize whitespace
        name = re.sub(r"\s{2,}", " ", name).strip()

    # Opposite party: try to detect Accused vs Complainant via keywords and sections
    oparty = None
    if re.search(r"\b(accused|आरोपी|प्रतिवादी)\b", t, re.IGNORECASE):
        # if labelled 'Accused' include that
        if re.search(r"(?:Accused|आरोपी|प्रतिवादी)[^\n]{0,80}", t, re.IGNORECASE):
            oparty = "Accused"
    if not oparty:
        # check if 'Complainant' or 'Informant' present
        if re.search(r"(?:Complainant|Informant|तक्रारदार|सूचक|खटील)", t, re.IGNORECASE):
            oparty = "Complainant"
    if not oparty:
        oparty = "Complainant"  # default assumption

    # Address extraction
    address = extract_address_by_label(t)
    if not address and locs:
        # create a compact address from 1-3 LOC tokens
        address = ", ".join(locs[:3]) if locs else None

    # If district missing but LOCs exist, use the first LOC as district
    if not district and locs:
        district = locs[0]

    # if police station missing but locs exist, search for pattern 'PS <loc>'
    if not police_station and locs:
        for l in locs:
            if re.search(rf"\b(?:PS|Police Station)\s+{re.escape(l)}\b", t, re.IGNORECASE):
                police_station = l
                break

    # Jurisdiction inference
    jurisdiction = None; jurisdiction_type = None
    if re.search(r"\bPAN[_\s-]?INDIA\b|ALL[_\s-]?INDIA", t, re.IGNORECASE):
        jurisdiction = "PAN_INDIA"; jurisdiction_type = "PAN_INDIA"
    elif district:
        jurisdiction = district; jurisdiction_type = "DISTRICT"
    elif state:
        jurisdiction = state; jurisdiction_type = "STATE"

    # Case category
    category = categorize_case(acts, sections, t)

    # Final canonical JSON
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
        "jurisdiction_type": jurisdiction_type,
    }

    # Final cleanup: ensure no huge blobs
    for k, v in out.items():
        if isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + "..."
    return out

# ---------------- Streamlit UI ----------------

st.title("FIR PII Extractor — Robust Hybrid (PDF + Paste Text)")
st.write(
    "Upload one or more FIR PDFs (scanned or digital) or paste FIR text. "
    "The pipeline uses strict regex + normalization first and NER as a fallback. "
    "Install Tesseract + Hindi/Marathi traineddata for best OCR results."
)

with st.sidebar:
    st.header("Settings")
    use_ner = st.checkbox("Use NER (Transformer pipeline) if available", value=NER_AVAILABLE)
    tesseract_langs = st.text_input("Tesseract languages", value="eng+hin+mar")
    st.markdown("**Note:** If you plan to use a custom fine-tuned NER model, modify `load_ner_pipeline()` in the code.")

uploaded_files = st.file_uploader("Upload FIR PDFs", type=["pdf"], accept_multiple_files=True)
raw_text = st.text_area("Or paste FIR text here", height=300, placeholder="Paste text extracted from Elasticsearch or copy-paste FIR text...")

ner_pipe = None
if use_ner and NER_AVAILABLE:
    ner_pipe = load_ner_pipeline()
    if ner_pipe is None:
        st.sidebar.warning("NER pipeline could not be loaded (offline or failed). Proceeding with regex-only.")

if st.button("Extract PII"):
    results: Dict[str, Any] = {}

    # Process uploaded files (if any)
    if uploaded_files:
        for uploaded in uploaded_files:
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                extracted_text = extract_text_from_pdf(tmp_path)
                if not extracted_text.strip():
                    st.warning(f"No text could be extracted from {uploaded.name}.")
                pii = extract_pii_from_text(extracted_text, use_ner=use_ner and ner_pipe is not None, ner_pipe=ner_pipe)
                results[uploaded.name] = pii
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Process pasted text (if present)
    if raw_text and raw_text.strip():
        pii = extract_pii_from_text(raw_text, use_ner=use_ner and ner_pipe is not None, ner_pipe=ner_pipe)
        results["pasted_text"] = pii

    if not results:
        st.warning("Please upload at least one PDF or paste FIR text to extract.")
    else:
        st.subheader("Extracted PII (cleaned output)")
        st.json(results, expanded=False)
        # Download button
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(payload.encode()).decode()
        href = f"data:application/json;base64,{b64}"
        st.markdown(f"[Download JSON]({href})")

        st.success("Extraction complete. Inspect fields and, if needed, fine-tune the NER model or add new normalization rules for your corpus.")

st.info("Tips: If outputs are noisy for scanned documents, try installing language packs for Tesseract (hin, mar). "
        "For production-grade performance, fine-tune a multilingual NER on labeled FIRs and/or use commercial OCR (Google Vision / AWS Textract).")
