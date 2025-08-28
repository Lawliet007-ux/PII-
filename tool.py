# tool.py
"""
Robust PII extractor for noisy OCR legal/FIR text (Hindi/English/mixed).
- No heavy ML models required (safe for Streamlit Cloud).
- Rule-first: regex + label parsing + heuristics.
- Outputs JSON list of extracted entities with label, text, confidence, and origin.
"""

import re
import json
import html
import streamlit as st
from difflib import SequenceMatcher

st.set_page_config(page_title="Robust FIR PII Extractor", layout="wide")


# ---------------------------
# Utilities / Normalization
# ---------------------------
def normalize_text(txt: str) -> str:
    if not txt:
        return txt
    t = txt
    # common OCR noise -> replace with spaces / normalized characters
    replacements = {
        "\u00A0": " ", "⁇": " ", "¡": " ", "‚": " ", "Ã": " ", "Ð": " ",
        "�": " ", "•": " ", "\ufeff": "", "\x0c": " ",
    }
    for k, v in replacements.items():
        t = t.replace(k, v)
    # replace weird combining marks and sequences of punctuation
    t = re.sub(r"[^\S\r\n]+", " ", t)           # normalize whitespace
    t = re.sub(r"[·•▪…]+", " ", t)
    t = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F]+", " ", t)
    # fix token subword markers (from HF outputs): "##word" -> "word"
    t = t.replace("##", "")
    # ensure consistent newlines
    t = re.sub(r"\r\n?", "\n", t)
    # remove control characters
    t = "".join(ch for ch in t if ord(ch) >= 9 and ord(ch) != 11 and ord(ch) != 12)
    t = t.strip()
    return t


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------
# Patterns (regex) for high-precision PII
# ---------------------------
REGEX_PATTERNS = {
    "phone": re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"),
    "date": re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),
    "time": re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b"),
    "fir_no": re.compile(r"\bFIR(?:\s*No\.?|\s*Number)?[:\s\-\)\.]*([A-Za-z0-9\/\-]{2,20})", re.IGNORECASE),
    "aadhaar": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "passport": re.compile(r"\b[A-Z]{1,2}\d{6,7}\b"),
    "pincode": re.compile(r"\b\d{6}\b"),
    "general_gd": re.compile(r"\b(?:General Diary Reference|G\.?D\.?\s*Ref|GD\s*Ref|G\.D\.? No)\b[:\s\-]*([A-Za-z0-9\/\-]+)?", re.IGNORECASE),
    "number": re.compile(r"\b\d{2,6}\b")
}

# ---------------------------
# Label keywords: mapping -> list of possible label tokens found in FIRs
# search is case-insensitive
# ---------------------------
LABEL_KEYWORDS = {
    "police_station": ["p.s.", "police station", "police thane", "polis thane", "ps", "police"],
    "district": ["district", "dist"],
    "fir_no": ["fir no", "fir number", "fir"],
    "date": ["date of", "date", "dated"],
    "time": ["time"],
    "complainant": ["complainant", "informant", "informant/complainant", "informant / complainant"],
    "accused": ["accused", "accused name"],
    "name": ["name", "name (name)"],
    "alias": ["alias"],
    "relative": ["son of", "s/o", "d/o", "relative", "father", "mother", "wife of", "husband of"],
    "address": ["address", "presented address", "address (p", "residence", "addr"],
    "uid": ["uid", "uid no", "aadhar", "aadhar no", "uid no."],
    "passport": ["passport", "passport no"],
    "dob": ["date of birth", "dob", "birth"],
    "phone": ["phone", "mobile", "mob", "phon"],
    "year": ["year"],
    "case_ref": ["general diary", "gd ref", "gd no", "general diary reference"],
}


# ---------------------------
# Helpers for line-based extraction
# ---------------------------
def split_lines(text: str):
    # split on blank lines, keep meaningful lines
    raw_lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
    # further split long lines on ' ; ' or ' . ' or ' , ' if they contain multiple labels
    lines = []
    for ln in raw_lines:
        # if line has multiple label indicators, break at '  ' sequences or ' ) ' followed by ':'
        sub = re.split(r'(?<=\))\s*[:\-]\s*|\s{2,}|\s*\|\s*', ln)
        # Keep original if split yields single piece
        if len(sub) == 1:
            lines.append(ln)
        else:
            for s in sub:
                s = s.strip()
                if s:
                    lines.append(s)
    return lines


def extract_value_after_colon(line: str) -> str:
    # Try common separators: ':', ')' followed by ':', ' - ', ' — '
    # Prefer last colon because labels may contain colon inside parentheses
    if ":" in line:
        parts = line.split(":", 1)
        # If left looks like label, right is value
        return parts[1].strip()
    # fallback: try parentheses ')'
    m = re.search(r'\)\s*(.*)$', line)
    if m:
        return m.group(1).strip()
    # fallback: last dash
    if "-" in line:
        parts = line.split("-", 1)
        return parts[1].strip()
    return ""


# ---------------------------
# Primary extraction functions
# ---------------------------
def label_based_extraction(lines):
    results = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        # Direct label:value style
        for label, keys in LABEL_KEYWORDS.items():
            for kw in keys:
                if kw in low:
                    val = extract_value_after_colon(ln)
                    # If no value after colon, maybe value on next line(s)
                    if not val:
                        # gather next up to 3 lines as possible value
                        following = []
                        for j in range(i + 1, min(i + 4, len(lines))):
                            # stop if next line contains another label keyword
                            nxt_low = lines[j].lower()
                            if any(k in nxt_low for klist in LABEL_KEYWORDS.values() for k in klist):
                                break
                            following.append(lines[j])
                        if following:
                            val = " ".join(following).strip()
                    if val:
                        results.append({
                            "label": label,
                            "text": clean_entity_text(val),
                            "confidence": 0.95,
                            "origin": "label"
                        })
                    break
    return results


def regex_all(text: str):
    results = []
    for lab, pat in REGEX_PATTERNS.items():
        for m in pat.finditer(text):
            # For FIR regex we captured group; handle accordingly
            if lab == "fir_no":
                g = m.group(1) if m.groups() else m.group(0)
                if g:
                    results.append({"label": "fir_no", "text": clean_entity_text(g), "confidence": 1.0, "origin": "regex"})
            elif lab == "general_gd":
                g = m.group(1) if m.groups() and m.group(1) else m.group(0)
                results.append({"label": "gd_ref", "text": clean_entity_text(g), "confidence": 0.95, "origin": "regex"})
            else:
                results.append({"label": lab, "text": clean_entity_text(m.group(0)), "confidence": 1.0, "origin": "regex"})
    return results


def find_dates_and_times(text: str):
    results = []
    # dates
    for m in REGEX_PATTERNS["date"].finditer(text):
        results.append({"label": "date", "text": clean_entity_text(m.group(0)), "confidence": 1.0, "origin": "regex"})
    # times
    for m in REGEX_PATTERNS["time"].finditer(text):
        results.append({"label": "time", "text": clean_entity_text(m.group(0)), "confidence": 1.0, "origin": "regex"})
    return results


def extract_devanagari_sequences(text: str):
    # Extract sequences of Devanagari chars (likely names/places)
    matches = re.findall(r'[\u0900-\u097F]{2,}(?:[\s,][\u0900-\u097F]{2,})*', text)
    out = []
    for m in matches:
        m = clean_entity_text(m)
        if len(m) >= 2:
            out.append({"label": "name_loc_hi", "text": m, "confidence": 0.85, "origin": "devanagari"})
    return out


def extract_latin_name_candidates(text: str):
    # Heuristic: contiguous sequences of Titlecase words (up to 4 words)
    pattern = re.compile(r'\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,}){0,3})\b')
    matches = pattern.findall(text)
    out = []
    for m in matches:
        m = clean_entity_text(m)
        # filter very common tokens that are not names
        if len(m) >= 3 and not re.match(r'^(Date|FIR|Police|Pune|India)$', m, re.I):
            out.append({"label": "name_loc_en", "text": m, "confidence": 0.6, "origin": "heuristic"})
    return out


def clean_entity_text(s: str) -> str:
    s = html.unescape(s)
    s = s.strip()
    # remove extra punctuation except commas and - and /
    s = re.sub(r'[\"\'\*\+\/\=\>\<\[\]\{\}\(\)\`~\|]+', '', s)
    # collapse multiple spaces
    s = re.sub(r'\s{2,}', ' ', s)
    # strip trailing punctuation
    s = s.strip(" ,;:-.")
    return s


def address_assembly(lines):
    """Find lines that look like addresses (contain city/district keywords or many commas)
       and assemble multi-line address blocks."""
    out = []
    address_indicators = ['address', 'presented address', 'addr', 'pune city', 'village', 'town', 'tehsil', 'taluka']
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(ind in low for ind in address_indicators) or (ln.count(",") >= 1 and len(ln) > 20):
            # gather this line plus following 2 lines if they look like continuation
            block = [ln]
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if nxt == "":
                    break
                if any(k in nxt.lower() for k in LABEL_KEYWORDS.keys()):
                    break
                # include continuation lines if they are not tiny
                if len(nxt) > 3:
                    block.append(nxt)
            addr = clean_entity_text(" ".join(block))
            out.append({"label": "address", "text": addr, "confidence": 0.9, "origin": "address_block"})
    return out


# ---------------------------
# Merge + dedupe
# ---------------------------
def dedupe_entities(entities):
    canonical = []
    seen_texts = set()
    for e in entities:
        text_norm = re.sub(r'\s+', ' ', e["text"].strip()).lower()
        if not text_norm:
            continue
        # skip extremely short junk unless high-confidence regex
        if len(text_norm) <= 1 and e.get("confidence", 0) < 0.95:
            continue
        # dedupe by exact/close similarity
        if any(similar(text_norm, other) > 0.92 for other in seen_texts):
            continue
        seen_texts.add(text_norm)
        canonical.append(e)
    return canonical


# ---------------------------
# Top-level orchestrator
# ---------------------------
def extract_pii_from_text(raw_text: str):
    txt = normalize_text(raw_text)

    # quick early return
    if not txt or txt.strip() == "":
        return []

    lines = split_lines(txt)

    entities = []

    # 1. Label-based extraction (high-priority)
    entities.extend(label_based_extraction(lines))

    # 2. Regex extraction across full text (IDs, dates, phone)
    entities.extend(regex_all(txt))
    entities.extend(find_dates_and_times(txt))

    # 3. Address assembly
    entities.extend(address_assembly(lines))

    # 4. Script-aware heuristics (Devanagari sequences)
    entities.extend(extract_devanagari_sequences(txt))

    # 5. Latin heuristics for TitleCase name candidates
    entities.extend(extract_latin_name_candidates(txt))

    # 6. If no complainant found, try to find lines containing 'Complainant' inside longer tokens
    # (already handled in label_based_extraction - keep as fallback)
    # 7. Normalize & dedupe
    merged = []

    # normalize labels: map some labels to broad categories
    label_map = {
        "name": "name",
        "complainant": "complainant",
        "accused": "accused",
        "relative": "relative",
        "alias": "alias",
        "police_station": "police_station",
        "district": "district",
        "address": "address",
        "uid": "uid",
        "passport": "passport",
        "fir_no": "fir_no",
        "date": "date",
        "time": "time",
        "phone": "phone",
        "pincode": "pincode",
        "pan": "pan",
        "aadhaar": "aadhaar",
        "name_loc_hi": "name_loc",
        "name_loc_en": "name_loc",
        "org": "org",
        "gd_ref": "gd_ref"
    }

    for e in entities:
        lab = e.get("label", "")
        lab_norm = label_map.get(lab, lab)
        merged.append({
            "label": lab_norm,
            "text": clean_entity_text(e.get("text", "")),
            "confidence": e.get("confidence", 0.5),
            "origin": e.get("origin", "heuristic")
        })

    final = dedupe_entities(merged)
    # sort by confidence descending
    final.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return final


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📑 Robust FIR/Legal PII Extractor (Rule-first, OCR-robust)")
st.markdown(
    "Paste noisy OCR / FIR text (Hindi/English / mixed). This tool uses deterministic rules + heuristics "
    "to extract IDs, dates, phones, police station, district, addresses and name candidates. "
    "It avoids heavy ML so it runs in constrained environments."
)

text_input = st.text_area("Paste FIR / legal text here (or upload a .txt):", height=360)
uploaded = st.file_uploader("Or upload a .txt file (optional)", type=["txt"])
if uploaded and (not text_input.strip()):
    try:
        raw = uploaded.read()
        try:
            text_input = raw.decode("utf-8")
        except:
            text_input = raw.decode("latin-1")
    except Exception as e:
        st.error("Failed to read uploaded file: " + str(e))

col1, col2 = st.columns([3, 1])
with col2:
    st.markdown("**Options**")
    show_debug = st.checkbox("Show extraction debug (line-by-line)", value=False)
    include_heuristics = st.checkbox("Include loose name heuristics (may add noisy candidates)", value=True)
    min_conf = st.slider("Minimum confidence to show", 0.0, 1.0, 0.4, 0.05)

if st.button("Extract PII"):
    if not text_input or text_input.strip() == "":
        st.warning("Please paste or upload some text first.")
    else:
        with st.spinner("Extracting — applying rules & heuristics..."):
            all_entities = extract_pii_from_text(text_input)

            # optionally filter out heuristics if disabled
            if not include_heuristics:
                all_entities = [e for e in all_entities if e["origin"] != "heuristic" and e["origin"] != "name_loc" and e["origin"] != "name_loc_hi"]

            # apply confidence filter
            displayed = [e for e in all_entities if e.get("confidence", 0) >= min_conf]

        if displayed:
            st.success(f"Found {len(displayed)} entity candidates (confidence >= {min_conf})")
            # pretty JSON
            st.json(displayed)
            st.markdown("Download results:")
            st.download_button("Download JSON", data=json.dumps(displayed, ensure_ascii=False, indent=2), file_name="pii_extraction.json", mime="application/json")
        else:
            st.info("No entities found with the current filters.")

        if show_debug:
            st.markdown("### Debug: normalized lines")
            st.code("\n".join(split_lines(normalize_text(text_input))))
            st.markdown("### Debug: raw normalized text (first 1200 chars)")
            st.code(normalize_text(text_input)[:1200])
