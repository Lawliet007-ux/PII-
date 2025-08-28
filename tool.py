import re
from rapidfuzz import fuzz

# ------------------- Regex Patterns -------------------
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"(?:FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+)",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "pin": r"\b\d{6}\b"
}

# ------------------- Gazetteer -------------------
GAZETTEER = {
    "state": ["Maharashtra", "Delhi", "Karnataka", "Goa"],
    "district": ["Pune", "Nagpur", "Mumbai", "Bhosari"],
    "police": ["Police Station", "PS", "Thane", "Bhosari Police Station"]
}

def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group().strip(), "confidence": 0.99})
    return out

def gazetteer_extract(text):
    out = []
    for label, words in GAZETTEER.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", text, flags=re.IGNORECASE):
                out.append({"label": label, "text": w, "confidence": 0.98})
    return out

def clean_ner_output(ner_results):
    """Merge subwords & filter junk"""
    cleaned = []
    buffer = []
    current_label = None
    
    for ent in ner_results:
        if ent["score"] < 0.75:  # discard weak predictions
            continue
        word = ent["word"].replace("##", "").strip()
        if len(word) <= 2:  # drop tiny fragments
            continue

        if ent["entity_group"] != current_label:
            if buffer:
                cleaned.append({
                    "label": current_label,
                    "text": " ".join(buffer),
                    "confidence": round(float(ent["score"]), 3)
                })
            buffer = [word]
            current_label = ent["entity_group"]
        else:
            buffer.append(word)

    if buffer:
        cleaned.append({
            "label": current_label,
            "text": " ".join(buffer),
            "confidence": 0.9
        })
    return cleaned

def merge_results(*sources):
    final = []
    for src in sources:
        for r in src:
            if not any(fuzz.ratio(r["text"].lower(), f["text"].lower()) > 90 for f in final):
                final.append(r)
    return final

def extract_pii(text, ner_model=None):
    regex_hits = regex_extract(text)
    gazette_hits = gazetteer_extract(text)
    ner_hits = []
    if ner_model:
        ents = ner_model(text)
        ner_hits = clean_ner_output(ents)
    return merge_results(regex_hits, gazette_hits, ner_hits)
