import re, json
import pandas as pd
from transformers import pipeline
from rapidfuzz import fuzz

# ---------------------
# Load NER Model
# ---------------------
ner_model = pipeline("ner", model="ai4bharat/IndicNER", aggregation_strategy="simple")

# ---------------------
# Regex Patterns
# ---------------------
REGEX_PATTERNS = {
    "aadhaar": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "phone": r"(?:(?:\+91|91|0)?[\-\s]?)?\d{10}\b",
    "email": r"[\w\.-]+@[\w\.-]+\.\w{2,}",
    "fir_no": r"FIR\s*(?:No\.?|Number)?[:\s]*[\w\d/-]+",
    "date": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    "time": r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d\b",
    "pin": r"\b\d{6}\b"
}

# ---------------------
# Gazetteer (expandable)
# ---------------------
DISTRICTS = ["Pune City", "Nagpur", "Mumbai", "Delhi", "Lucknow"]  # demo
POLICE_TERMS = ["Police Station", "P.S.", "Thane", "Thana"]

# ---------------------
# Extractors
# ---------------------
def regex_extract(text):
    out = []
    for label, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            out.append({"label": label, "text": m.group(), "confidence": 0.95})
    return out

def ner_extract(text):
    results = []
    words = text.split()
    chunk_size = 200
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        ents = ner_model(chunk)
        for e in ents:
            results.append({
                "label": e["entity_group"],
                "text": e["word"],
                "confidence": float(e["score"])
            })
    return results

def rule_extract(text):
    out = []
    # Police Station
    ps_match = re.findall(r"(?:P\.S\.|Police Thane|Police Station)[:\-]?\s*([A-Za-z\u0900-\u097F ]+)", text)
    for ps in ps_match:
        out.append({"label": "POLICE_STATION", "text": ps.strip(), "confidence": 0.9})
    # District
    dist_match = re.findall(r"District[:\-]?\s*([A-Za-z\u0900-\u097F ]+)", text)
    for d in dist_match:
        out.append({"label": "DISTRICT", "text": d.strip(), "confidence": 0.9})
    return out

def gazetteer_match(text):
    out = []
    for d in DISTRICTS:
        if d.lower() in text.lower():
            out.append({"label": "DISTRICT", "text": d, "confidence": 0.85})
    return out

# ---------------------
# Merge + Deduplicate
# ---------------------
def merge_results(results):
    final, seen = [], set()
    for r in results:
        key = r["text"].lower()
        if not any(fuzz.ratio(key, f["text"].lower()) > 90 and r["label"] == f["label"] for f in final):
            final.append(r)
    return final

# ---------------------
# Main
# ---------------------
def extract_pii(text: str):
    regex_hits = regex_extract(text)
    ner_hits = ner_extract(text)
    rule_hits = rule_extract(text)
    gaz_hits = gazetteer_match(text)

    all_hits = regex_hits + ner_hits + rule_hits + gaz_hits
    return merge_results(all_hits)
