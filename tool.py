# tool.py
# Robust PII extractor for noisy OCR FIR/legal text (Hindi/English/mixed)
# - Rule-first (regex + label parsing + block assembly)
# - Optional multilingual NER via transformers with safe chunking
# - Confidence scoring and strict noise filtering to avoid junk
# - Runs on Streamlit Cloud; if ML can't load, it falls back cleanly

import re
import json
import html
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

import streamlit as st

st.set_page_config(page_title="📑 Robust FIR PII Extractor", layout="wide")


# =========================
# Optional: Transformers NER
# =========================
@st.cache_resource(show_spinner=False)
def load_hf():
    """Try to load a multilingual NER model; return (pipe, available)."""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        model_id = "Davlan/xlm-roberta-base-ner-hrl"  # covers hi/en reasonably well
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForTokenClassification.from_pretrained(model_id)
        ner = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",  # groups subword tokens
            device=-1
        )
        return ner, True
    except Exception as e:
        # If downloads are blocked or libs missing, continue without ML
        return None, False


NER_PIPE, HF_AVAILABLE = load_hf()


# ================
# Normalization
# ================
def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    repl = {
        "\u00A0": " ", "⁇": " ", "Ð": " ", "ã": "a", "¡": " ", "�": " ",
        "\ufeff": "", "\x0c": " ", "•": " ", "·": " ", "▪": " "
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    # remove "##" subword markers & odd punctuation clusters
    t = t.replace("##", "")
    # newline normalize
    t = re.sub(r"\r\n?", "\n", t)
    # collapse whitespace
    t = re.sub(r"[^\S\r\n]+", " ", t)
    # trim weird punctuation
    t = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F]+", " ", t)
    t = t.strip()
    return t


def split_lines(text: str) -> List[str]:
    """Keep meaningful lines; also split long lines that likely contain multiple labels."""
    raw = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
    out = []
    for ln in raw:
        # micro-splits to isolate label:value pairs
        parts = re.split(r"(?<=\))\s*[:\-]\s*|\s{2,}|\s*\|\s*", ln)
        if len(parts) == 1:
            out.append(ln)
        else:
            for p in parts:
                p = p.strip()
                if p:
                    out.append(p)
    return out


def clean_entity_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = s.strip()
    # remove extra punctuation except , - /
    s = re.sub(r"[\"\'\*\=\>\<\[\]\{\}\(\)\`~\|]+", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = s.strip(" ,;:-.")
    return s


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ================
# Regex patterns
# ================
PAT = {
    "phone": re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "date_dmy": re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),
    "time": re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b"),
    "fir": re.compile(r"FIR\s*No\.?\s*(?:\([^)]+\))?\s*[:\-]?\s*([A-Za-z0-9\/\-]{1,30})", re.I),
    "dob": re.compile(r"Date\s*\/?\s*Year\s*of\s*Birth[^0-9]*([0-3]?\d[\/\-][01]?\d[\/\-]\d{2,4})", re.I),
    "gd": re.compile(r"(General\s*Diary\s*Reference|G\.?D\.?(?:\s*No\.?)?|GD\s*Ref)[^A-Za-z0-9]{0,10}([A-Za-z0-9\/\-]{1,20})", re.I),
    "aadhaar": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,7}\b"),
    "pincode": re.compile(r"\b\d{6}\b"),
    "currency": re.compile(r"\b(?:₹\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{2})\b"),
    "section": re.compile(r"\b(?:Section|Sec\.?)\s*([0-9]{1,4}[A-Za-z]?(?:\s*\(\s*[a-z]\s*\))?)", re.I),
}

LABEL_KEYS = {
    "police_station": ["p.s.", "police station", "police thane", "polis thane", "police tha", "ps"],
    "district": ["district", "dist"],
    "nationality": ["nationality"],
    "address": ["address", "presented address", "address type", "residence", "addr"],
    "complainant": ["complainant", "informant"],
    "alias": ["alias"],
    "relative": ["s/o", "d/o", "w/o", "son of", "daughter of", "wife of", "husband of"],
    "fir_no": ["fir no", "fir number", "fir"],
    "dob": ["date/year of birth", "date of birth", "dob", "birth"],
    "gd": ["general diary reference", "gd ref", "gd no", "g.d."],
    "date": ["date and time of fir", "date", "dated"],
    "time": ["time"],
}


# =======================
# Core extraction steps
# =======================
def regex_extract(text: str) -> List[Dict[str, Any]]:
    out = []

    # Phones
    for m in PAT["phone"].finditer(text):
        out.append({"label": "phone", "text": m.group(0), "confidence": 1.0, "origin": "regex"})

    # Dates/times
    for m in PAT["date_dmy"].finditer(text):
        out.append({"label": "date", "text": m.group(0), "confidence": 1.0, "origin": "regex"})
    for m in PAT["time"].finditer(text):
        out.append({"label": "time", "text": m.group(0).replace(".", ":"), "confidence": 1.0, "origin": "regex"})

    # FIR No
    m = PAT["fir"].search(text)
    if m and m.group(1):
        out.append({"label": "fir_no", "text": clean_entity_text(m.group(1)), "confidence": 0.99, "origin": "regex"})

    # DOB
    m = PAT["dob"].search(text)
    if m:
        out.append({"label": "dob", "text": m.group(1), "confidence": 0.99, "origin": "regex"})

    # GD Reference
    m = PAT["gd"].search(text)
    if m:
        out.append({"label": "gd_ref", "text": clean_entity_text(m.group(2)), "confidence": 0.95, "origin": "regex"})

    # IDs
    for lab in ["aadhaar", "pan", "passport", "pincode"]:
        for m in PAT[lab].finditer(text):
            out.append({"label": lab, "text": m.group(0), "confidence": 0.98, "origin": "regex"})

    # Sections
    for m in PAT["section"].finditer(text):
        out.append({"label": "offence_section", "text": clean_entity_text(m.group(1)), "confidence": 0.9, "origin": "regex"})

    # Currency values
    for m in PAT["currency"].finditer(text):
        out.append({"label": "value_inr", "text": m.group(0), "confidence": 0.9, "origin": "regex"})

    return out


def value_after_colon(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    m = re.search(r"\)\s*(.*)$", line)
    if m:
        return m.group(1).strip()
    if "-" in line:
        return line.split("-", 1)[1].strip()
    return ""


def label_parse(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    L = len(lines)
    # Scan each line for label keys; collect value in same or next few lines
    for i, ln in enumerate(lines):
        low = ln.lower()
        for label, keys in LABEL_KEYS.items():
            if any(k in low for k in keys):
                val = value_after_colon(ln)
                if not val:
                    # collect next lines until hitting another label-y line or blank
                    follow = []
                    for j in range(i + 1, min(i + 5, L)):
                        nxt = lines[j].strip()
                        if not nxt:
                            break
                        nxt_low = nxt.lower()
                        if any(k in nxt_low for kk in LABEL_KEYS.values() for k in kk):
                            break
                        follow.append(nxt)
                    val = " ".join(follow).strip()
                if val:
                    lab_norm = {
                        "fir_no": "fir_no",
                        "dob": "dob",
                        "gd": "gd_ref",
                    }.get(label, label)
                    out.append({
                        "label": lab_norm,
                        "text": clean_entity_text(val),
                        "confidence": 0.95,
                        "origin": "label"
                    })
    # Special case: Police Station often like "P.S. (Police Thane): Bhosari"
    for ln in lines:
        m = re.search(r"(?:P\.?S\.?|Police\s*(?:Station|Th[ae]ne))[^:]*:\s*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s\-]+)", ln, re.I)
        if m:
            out.append({"label": "police_station", "text": clean_entity_text(m.group(1)), "confidence": 0.98, "origin": "label"})
        m2 = re.search(r"District[^:]*:\s*([A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\s\-]+)", ln, re.I)
        if m2:
            out.append({"label": "district", "text": clean_entity_text(m2.group(1)), "confidence": 0.98, "origin": "label"})
    return out


def assemble_addresses(lines: List[str]) -> List[Dict[str, Any]]:
    out = []
    indicators = ["address", "presented address", "residence", "addr"]
    L = len(lines)
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(ind in low for ind in indicators) or ("," in ln and len(ln) > 18):
            block = [ln]
            for j in range(i + 1, min(i + 5, L)):
                nxt = lines[j].strip()
                if not nxt:
                    break
                nxt_low = nxt.lower()
                if any(k in nxt_low for kk in LABEL_KEYS.values() for k in kk):
                    break
                block.append(nxt)
            addr = clean_entity_text(" ".join(block))
            # Heuristic gate to avoid junk
            if len(addr) >= 15 and not re.fullmatch(r"[A-Za-z]{1,3}", addr):
                out.append({"label": "address", "text": addr, "confidence": 0.9, "origin": "address_block"})
    return out


# ==============
# ML NER (opt.)
# ==============
def chunk_text(text: str, max_chars: int = 900, overlap: int = 80) -> List[Tuple[int, str]]:
    """Yield (start_index, chunk) with overlap to avoid boundary misses."""
    chunks = []
    n = len(text)
    step = max_chars - overlap
    i = 0
    while i < n:
        chunk = text[i:i + max_chars]
        chunks.append((i, chunk))
        i += step
    return chunks


def ner_extract(text: str, min_score: float = 0.85) -> List[Dict[str, Any]]:
    if not HF_AVAILABLE or not NER_PIPE:
        return []
    ents = []
    for start, chunk in chunk_text(text):
        res = NER_PIPE(chunk)
        for r in res:
            if r.get("score", 0) < min_score:
                continue
            lab = r.get("entity_group", "").upper()
            if lab not in {"PER", "ORG", "LOC"}:
                continue
            span = clean_entity_text(chunk[r["start"]:r["end"]])
            # strict noise filters
            if not span or len(span) < 3:
                continue
            if re.fullmatch(r"[A-Za-z]\.?$", span):
                continue
            if re.fullmatch(r"[0-9]+$", span):
                continue
            if span.lower() in {"police", "station", "district", "india"}:
                # avoid generic labels from headers
                continue
            ents.append({
                "label": {"PER": "person", "ORG": "org", "LOC": "loc"}[lab],
                "text": span,
                "confidence": float(r["score"]),
                "origin": "ml"
            })
    return ents


# ==================
# Merge & dedupe
# ==================
def dedupe(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = []
    for e in entities:
        txt = re.sub(r"\s+", " ", e["text"]).strip().lower()
        if not txt:
            continue
        # skip ultra-short unless very high confidence and structured
        if len(txt) <= 2 and e.get("origin") != "regex":
            continue
        if any(similar(txt, re.sub(r"\s+", " ", s["text"]).strip().lower()) > 0.92 and s["label"] == e["label"] for s in out):
            continue
        out.append(e)
    # sort by label then confidence desc
    out.sort(key=lambda x: (x["label"], -x.get("confidence", 0)))
    return out


# ==================
# Orchestrator
# ==================
def extract_all(raw: str, use_ml: bool = True) -> List[Dict[str, Any]]:
    txt = normalize_text(raw)
    if not txt:
        return []
    lines = split_lines(txt)

    entities = []
    entities += regex_extract(txt)
    entities += label_parse(lines)
    entities += assemble_addresses(lines)

    if use_ml:
        entities += ner_extract(txt, min_score=0.86)

    # Normalize some fields
    for e in entities:
        if e["label"] == "time":
            e["text"] = e["text"].replace(".", ":")
        if e["label"] in {"date", "dob"}:
            # try to normalize to dd/mm/yyyy -> yyyy-mm-dd (best-effort)
            m = re.match(r"^(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})$", e["text"])
            if m:
                d, mth, y = m.groups()
                if len(y) == 2:
                    y = "20" + y
                e["text"] = f"{y.zfill(4)}-{mth.zfill(2)}-{d.zfill(2)}"

    entities = dedupe(entities)
    return entities


# ==================
# Streamlit UI
# ==================
st.title("📑 Robust FIR / Legal PII Extractor (Hindi/English/Mixed)")
st.caption("Deterministic rules + optional multilingual NER. No junk; high precision; chunking-safe.")

left, right = st.columns([3, 1])
with left:
    raw_text = st.text_area("Paste OCR/FIR text:", height=360, placeholder="Paste noisy OCR text here...")
    file_up = st.file_uploader("Or upload a .txt file", type=["txt"])
    if file_up and not raw_text.strip():
        try:
            raw = file_up.read()
            try:
                raw_text = raw.decode("utf-8")
            except:
                raw_text = raw.decode("latin-1")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

with right:
    st.markdown("**Options**")
    use_ml = st.checkbox("Use multilingual NER (transformers)", value=True if HF_AVAILABLE else False,
                         help="If disabled or unavailable, only rule-based extraction is used.")
    min_conf = st.slider("Minimum confidence to display", 0.0, 1.0, 0.40, 0.05)
    show_debug = st.checkbox("Show normalized text & lines", value=False)

run = st.button("🔍 Extract PII", use_container_width=True)

if run:
    if not raw_text.strip():
        st.warning("Please paste or upload text first.")
    else:
        try:
            with st.spinner("Extracting…"):
                results = extract_all(raw_text, use_ml=use_ml and HF_AVAILABLE)
                # apply threshold
                results = [r for r in results if r.get("confidence", 0) >= min_conf]

            st.subheader("📑 Extracted PII")
            if not results:
                st.info("No entities found with current threshold. Try lowering it or enabling ML.")
            else:
                st.json(results)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(results, ensure_ascii=False, indent=2),
                    file_name="pii_extraction.json",
                    mime="application/json",
                    use_container_width=True
                )

            if show_debug:
                st.markdown("### 🔧 Debug: normalized text (first 1200 chars)")
                st.code(normalize_text(raw_text)[:1200])
                st.markdown("### 🔧 Debug: split lines")
                st.code("\n".join(split_lines(normalize_text(raw_text))[:120]))

        except Exception as e:
            st.exception(e)
            st.error("Extraction failed. The app fell back gracefully — check the logs for details.")
            # Try rule-only fallback once
            try:
                res2 = extract_all(raw_text, use_ml=False)
                res2 = [r for r in res2 if r.get("confidence", 0) >= min_conf]
                st.info("Rule-based fallback results:")
                st.json(res2)
            except Exception as e2:
                st.exception(e2)
                st.error("Rule-based fallback also failed.")
