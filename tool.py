# fir_pii_extractor_enhanced.py
"""
Enhanced FIR PII Extractor with improved extraction patterns and logic
- Better regex patterns covering more FIR formats
- Multi-stage extraction with fallbacks
- Improved context-aware extraction
- Better handling of Hindi/English mixed content
"""

import streamlit as st
import fitz
import pdfplumber
import pytesseract
from PIL import Image
import re, os, json, unicodedata, tempfile, base64
from typing import List, Dict, Any, Optional, Tuple, Set
from rapidfuzz import process, fuzz
from collections import Counter
import datetime

# Optional transformers NER
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="Enhanced FIR PII Extractor", layout="wide")

# ---------------- Enhanced Constants ----------------
SECTION_MAX = 999
SECTION_MIN_KEEP = 1

# Expanded placeholders to filter out
PLACEHOLDERS = set([
    "name", "नाव", "नाम", "type", "address", "of p.s.", "then name of p.s.", "of p.s", "of ps",
    "complainant", "informant", "accused", "victim", "witness", "father", "son", "daughter",
    "police station", "district", "state", "pin code", "mobile", "phone", "email",
    "date", "time", "place", "incident", "offence", "section", "act", "fir", "cr"
])

# Enhanced state mapping with variations
STATE_VARIATIONS = {
    "maharashtra": ["Maharashtra", "महाराष्ट्र", "mh", "maharastra"],
    "uttar pradesh": ["Uttar Pradesh", "उत्तर प्रदेश", "up", "u.p.", "uttar predesh"],
    "delhi": ["Delhi", "दिल्ली", "new delhi", "ncr"],
    "karnataka": ["Karnataka", "कर्नाटक", "ka", "karnatak"],
    "gujarat": ["Gujarat", "गुजरात", "gj", "gujrat"],
    "bihar": ["Bihar", "बिहार", "br"],
    "tamil nadu": ["Tamil Nadu", "तमिल नाडु", "tn", "tamilnadu"],
    "rajasthan": ["Rajasthan", "राजस्थान", "rj"],
    "madhya pradesh": ["Madhya Pradesh", "मध्य प्रदेश", "mp", "m.p."],
    "west bengal": ["West Bengal", "पश्चिम बंगाल", "wb", "bengal"],
    "andhra pradesh": ["Andhra Pradesh", "आंध्र प्रदेश", "ap", "a.p."],
    "telangana": ["Telangana", "तेलंगाना", "ts"],
    "kerala": ["Kerala", "केरल", "kl"],
    "odisha": ["Odisha", "ओडिशा", "or", "orissa"],
    "punjab": ["Punjab", "पंजाब", "pb"],
    "haryana": ["Haryana", "हरियाणा", "hr"],
    "assam": ["Assam", "असम", "as"],
    "jharkhand": ["Jharkhand", "झारखंड", "jh"],
    "uttarakhand": ["Uttarakhand", "उत्तराखंड", "uk", "uttaranchal"],
    "himachal pradesh": ["Himachal Pradesh", "हिमाचल प्रदेश", "hp", "h.p."],
    "jammu and kashmir": ["Jammu and Kashmir", "जम्मू और कश्मीर", "jk", "j&k"],
    "goa": ["Goa", "गोवा", "ga"],
    "chhattisgarh": ["Chhattisgarh", "छत्तीसगढ़", "cg", "chattisgarh"]
}

# Enhanced district seeds with more cities
DISTRICT_SEED = [
    # Maharashtra
    "Pune", "Mumbai", "Nagpur", "Nashik", "Aurangabad", "Solapur", "Amravati", "Kolhapur",
    "Sangli", "Ahmednagar", "Latur", "Dhule", "Nanded", "Jalgaon", "Akola", "Satara",
    # UP
    "Lucknow", "Kanpur", "Ghaziabad", "Agra", "Meerut", "Varanasi", "Allahabad", "Bareilly",
    "Moradabad", "Aligarh", "Gorakhpur", "Saharanpur", "Noida", "Firozabad", "Jhansi",
    # Delhi
    "New Delhi", "Central Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi",
    # Other major cities
    "Bengaluru", "Chennai", "Hyderabad", "Ahmedabad", "Kolkata", "Surat", "Jaipur", "Indore",
    "Bhopal", "Patna", "Visakhapatnam", "Vadodara", "Coimbatore", "Kochi", "Thiruvananthapuram"
]

# Enhanced police station patterns
POLICE_STATION_COMMON = [
    "City", "Rural", "Cyber", "Traffic", "Women", "Economic Offences", "Crime Branch",
    "Special Branch", "Anti Corruption", "Narcotics", "Airport", "Railway"
]

# Enhanced acts dictionary with variations
ACTS_MAPPING = {
    # IPC variations
    "ipc": "Indian Penal Code 1860",
    "indian penal code": "Indian Penal Code 1860", 
    "penal code": "Indian Penal Code 1860",
    "भारतीय दंड संहिता": "Indian Penal Code 1860",
    
    # CrPC variations
    "crpc": "Code of Criminal Procedure 1973",
    "code of criminal procedure": "Code of Criminal Procedure 1973",
    "criminal procedure code": "Code of Criminal Procedure 1973",
    "cr.p.c": "Code of Criminal Procedure 1973",
    
    # IT Act variations
    "information technology": "Information Technology Act 2000",
    "it act": "Information Technology Act 2000",
    "cyber": "Information Technology Act 2000",
    "सूचना प्रौद्योगिकी अधिनियम": "Information Technology Act 2000",
    
    # Other acts
    "arms act": "Arms Act 1959",
    "ndps": "Narcotic Drugs and Psychotropic Substances Act 1985",
    "pocso": "Protection of Children from Sexual Offences Act 2012",
    "dowry": "Dowry Prohibition Act 1961",
    "domestic violence": "Protection of Women from Domestic Violence Act 2005",
    "sc st": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act 1989",
    "motor vehicle": "Motor Vehicles Act 1988",
    "prevention of corruption": "Prevention of Corruption Act 1988",
    "explosives": "Explosives Act 1884",
    "passport": "Passport Act 1967",
    "customs": "Customs Act 1962",
    "foreign exchange": "Foreign Exchange Management Act 1999"
}

# Enhanced case categories
CASE_CATEGORIES = {
    "MURDER": ["302", "300", "299"],
    "THEFT": ["378", "379", "380", "381", "382"],
    "ROBBERY": ["392", "393", "394", "395", "396", "397", "398", "399", "400", "401", "402"],
    "SEXUAL_OFFENCE": ["354", "354a", "354b", "354c", "354d", "375", "376", "377", "509"],
    "ASSAULT": ["322", "323", "324", "325", "326", "351", "352"],
    "FRAUD": ["406", "407", "408", "409", "415", "416", "417", "418", "419", "420"],
    "CYBER_CRIME": ["66", "66a", "66b", "66c", "66d", "66e", "66f", "67", "67a", "67b"],
    "DOWRY": ["304b", "498a"],
    "KIDNAPPING": ["363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373"],
    "EXTORTION": ["383", "384", "385", "386", "387", "388", "389"],
    "CRIMINAL_INTIMIDATION": ["503", "504", "505", "506", "507", "508"],
    "HURT": ["319", "320", "321", "322", "323", "324", "325", "326"],
    "CHEATING": ["415", "416", "417", "418", "419", "420"],
    "CRIMINAL_BREACH_OF_TRUST": ["405", "406", "407", "408", "409"],
    "MISCHIEF": ["425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
    "TRESPASS": ["441", "442", "443", "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458", "459", "460"]
}

# ---------------- Enhanced Utilities ----------------

def dedupe_preserve_order(items: List[str]) -> List[str]:
    if not items:
        return []
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def remove_control_chars(s: str) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

def strip_nonessential_unicode(s: str) -> str:
    if not s:
        return ""
    # Keep ASCII, Devanagari, common punctuation, and currency symbols
    return re.sub(r"[^\x00-\x7F\u0900-\u097F\u2000-\u206F\u20B9\n\t:;.,/()\-—'\"°]", " ", s)

def collapse_spaces(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s.strip()

def fix_broken_devanagari_runs(s: str) -> str:
    if not s:
        return ""
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

def is_placeholder_text(text: str) -> bool:
    if not text:
        return True
    
    text_clean = re.sub(r'\W+', ' ', text.lower().strip())
    if len(text_clean) < 2:
        return True
        
    # Check against known placeholders
    if text_clean in PLACEHOLDERS:
        return True
    
    # Check for common placeholder patterns
    placeholder_patterns = [
        r'^(name|type|address)$',
        r'^(complainant|informant|accused)$',
        r'^\w{1,3}$',  # Very short words
        r'^[._\-\s]+$',  # Only punctuation/spaces
        r'^\d{1,2}$',  # Single/double digits only
    ]
    
    for pattern in placeholder_patterns:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True
    
    return False

def clean_and_validate_field(text: str, max_length: int = 500) -> Optional[str]:
    if not text:
        return None
    
    cleaned = text.strip()
    if is_placeholder_text(cleaned):
        return None
    
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned

# ---------------- Enhanced Text Extraction ----------------

def extract_text_pymupdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text)
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"PyMuPDF extraction failed: {str(e)}")
        return ""

def extract_text_pdfplumber(path: str) -> str:
    try:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except Exception as e:
        st.warning(f"pdfplumber extraction failed: {str(e)}")
        return ""

def extract_text_ocr(path: str, tesseract_langs: str) -> str:
    try:
        doc = fitz.open(path)
        ocr_parts = []
        for page in doc:
            # Increase resolution for better OCR
            mat = fitz.Matrix(3, 3)  # 3x zoom
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR with custom config for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u0900-\u097F .,/-:()'
            text = pytesseract.image_to_string(img, lang=tesseract_langs, config=custom_config)
            if text:
                ocr_parts.append(text)
        doc.close()
        return "\n".join(ocr_parts)
    except Exception as e:
        st.warning(f"OCR extraction failed: {str(e)}")
        return ""

def extract_text_from_pdf(path: str, tesseract_langs: str = "eng+hin+mar") -> str:
    # Try PyMuPDF first
    text = extract_text_pymupdf(path)
    text = canonicalize_text(text)
    
    # If insufficient text, try pdfplumber
    if len(text) < 200:
        alt_text = extract_text_pdfplumber(path)
        alt_text = canonicalize_text(alt_text)
        if len(alt_text) > len(text):
            text = alt_text
    
    # If still insufficient, try OCR
    if len(text) < 200:
        ocr_text = extract_text_ocr(path, tesseract_langs)
        ocr_text = canonicalize_text(ocr_text)
        if len(ocr_text) > len(text):
            text = ocr_text
    
    return canonicalize_text(text)

# ---------------- Enhanced NER ----------------

@st.cache_resource
def load_ner_pipe():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        # Try Indic NER first
        try:
            return pipeline("token-classification", model="ai4bharat/indic-ner", aggregation_strategy="simple")
        except:
            # Fallback to multilingual NER
            return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")
    except Exception as e:
        st.warning(f"NER model loading failed: {str(e)}")
        return None

# ---------------- Enhanced Field Extractors ----------------

def find_fir_number_candidates(text: str) -> List[str]:
    candidates = []
    
    # Enhanced FIR number patterns
    fir_patterns = [
        r"(?:FIR\s*No\.?\s*|CR\s*No\.?\s*|Case\s*No\.?\s*|Registration\s*No\.?\s*|R\.?C\.?\s*No\.?\s*|आर\.?\s*सी\.?\s*|एफ\.?\s*आई\.?\s*आर\.?)\s*[:\-]?\s*([A-Za-z0-9/\-\s]{3,25})",
        r"(?:Crime\s*No\.?\s*|अपराध\s*संख्या)\s*[:\-]?\s*([A-Za-z0-9/\-\s]{3,25})",
        r"\b(\d{1,4}/\d{2,4})\b",  # Pattern like 123/2023
        r"\b([A-Za-z]{2,4}\s*\d{1,6}/\d{2,4})\b",  # Pattern like PS 123/2023
        r"(?:u/s|under section|धारा).*?(\d{1,4}/\d{2,4})"  # FIR number near sections
    ]
    
    for pattern in fir_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.group(1).strip()
            if len(candidate) >= 3:
                candidates.append(candidate)
    
    return dedupe_preserve_order(candidates)

def find_year_candidates(text: str) -> List[str]:
    candidates = []
    current_year = datetime.datetime.now().year
    
    # Year patterns
    year_patterns = [
        r"(?:Year|वर्ष|साल|Date.*?FIR|FIR.*?Date|दिनांक)\s*[:\-]?\s*(20\d{2}|19\d{2})",
        r"\b(20[0-2]\d|19[5-9]\d)\b",  # Years from 1950-2029
        r"(\d{1,2})[\/\-\.](0?\d|1[0-2])[\/\-\.]?(20\d{2}|19\d{2})",  # Date formats
        r"(20\d{2}|19\d{2})[\/\-\.]?(0?\d|1[0-2])[\/\-\.]?(\d{1,2})"   # Date formats
    ]
    
    for pattern in year_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extract year from different groups
            for group in match.groups():
                if group and len(group) == 4 and group.isdigit():
                    year = int(group)
                    if 1950 <= year <= current_year + 5:  # Reasonable year range
                        candidates.append(group)
    
    # Prioritize recent years
    if candidates:
        year_counts = Counter(candidates)
        candidates = [year for year, _ in year_counts.most_common()]
    
    return dedupe_preserve_order(candidates)

def find_state_candidates(text: str) -> List[str]:
    candidates = []
    
    # Direct state name matching
    for state_key, variations in STATE_VARIATIONS.items():
        for variation in variations:
            pattern = r'\b' + re.escape(variation) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                candidates.append(variations[0])  # Use canonical name
                break
    
    # State in address/jurisdiction patterns
    state_patterns = [
        r"(?:State|राज्य|प्रांत)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s]{4,25})",
        r"(?:District|Dist\.?\s*|जिला|जिल्हा).*?,\s*([A-Za-z\u0900-\u097F\s]{4,25})",
        r"\b([A-Za-z\u0900-\u097F\s]{4,25})\s+(?:Police|State|राज्य)"
    ]
    
    for pattern in state_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            candidate = match.group(1).strip()
            # Check if candidate matches any state
            for state_key, variations in STATE_VARIATIONS.items():
                for variation in variations:
                    if variation.lower() in candidate.lower():
                        candidates.append(variations[0])
                        break
    
    return dedupe_preserve_order(candidates)

def find_district_candidates(text: str) -> List[str]:
    candidates = []
    
    # Enhanced district patterns
    district_patterns = [
        r"(?:District|Dist\.?\s*|जिला|जिल्हा|District\s*Name)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9\s,./-]{2,50})",
        r"District\s*\([^)]*\)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s]{2,50})",
        r"P\.?S\.?\s*([A-Za-z\u0900-\u097F\s]{2,50})\s*(?:Dist\.?|District)",
        r"(?:Under|u/s).*?([A-Za-z\u0900-\u097F\s]{3,25})\s+(?:District|Dist\.)",
        r"\b([A-Za-z\u0900-\u097F\s]{3,25})\s+(?:City|Rural|Urban)\s*(?:Police|District)"
    ]
    
    for pattern in district_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.group(1).strip()
            # Clean up common suffixes/prefixes
            candidate = re.sub(r'\b(?:City|Rural|Urban|Police|District|Dist\.?)\b', '', candidate, flags=re.IGNORECASE).strip()
            if len(candidate) >= 3 and not is_placeholder_text(candidate):
                candidates.append(candidate)
    
    # Fuzzy match against known districts
    enhanced_candidates = []
    for candidate in candidates:
        best_match = process.extractOne(candidate, DISTRICT_SEED, scorer=fuzz.WRatio)
        if best_match and best_match[1] >= 70:
            enhanced_candidates.append(best_match[0])
        else:
            enhanced_candidates.append(candidate)
    
    return dedupe_preserve_order(enhanced_candidates)

def find_police_station_candidates(text: str) -> List[str]:
    candidates = []
    
    # Enhanced police station patterns
    ps_patterns = [
        r"(?:Police\s*Station|P\.?S\.?\s*|PS\b|पोलीस\s*ठाणे|पुलिस\s*थाना|थाना)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9\s.,-]{2,50})",
        r"(?:Station|Station\s*Name|थाना\s*नाम)\s*[:\-]?\s*([A-Za-z\u0900-\u097F0-9\s.,-]{2,50})",
        r"(?:Transferred\s*to|Transfer\s*to)\s*P\.?S\.?\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s]{2,50})",
        r"Crime\s*registered\s*at\s*([A-Za-z\u0900-\u097F\s]{2,50})\s*(?:Police|P\.S\.)",
        r"FIR.*?(?:at|@)\s*([A-Za-z\u0900-\u097F\s]{2,50})\s*(?:Police|P\.S\.)",
        r"\b([A-Za-z\u0900-\u097F\s]{2,50})\s+(?:Police\s*Station|P\.S\.)"
    ]
    
    for pattern in ps_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.group(1).strip()
            # Clean up
            candidate = re.sub(r'\b(?:Police|Station|P\.?S\.?)\b', '', candidate, flags=re.IGNORECASE).strip()
            if len(candidate) >= 2 and not is_placeholder_text(candidate):
                candidates.append(candidate)
    
    return dedupe_preserve_order(candidates)

def find_acts_candidates(text: str) -> List[str]:
    candidates = []
    
    # Direct act name matching
    for act_key, full_name in ACTS_MAPPING.items():
        pattern = r'\b' + re.escape(act_key) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            candidates.append(full_name)
    
    # Context-based act detection
    act_patterns = [
        r"(?:under|u/s|u/s\.|धारा|अंतर्गत).*?([A-Za-z\s]+(?:Act|अधिनियम|कायदा).*?(?:\d{4})?)",
        r"([A-Za-z\s]+(?:Act|अधिनियम|कायदा)\s*(?:\d{4})?)",
        r"(?:registered|दर्ज).*?([A-Za-z\s]+Act\s*\d{4})",
        r"(?:IPC|IT|NDPS|POCSO|Arms)\b",
        r"(?:Indian\s*Penal\s*Code|Information\s*Technology|Cyber)"
    ]
    
    for pattern in act_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            candidate = match.group(0).strip().lower()
            for act_key, full_name in ACTS_MAPPING.items():
                if act_key in candidate:
                    candidates.append(full_name)
    
    return dedupe_preserve_order(candidates)

def find_sections_candidates(text: str) -> List[str]:
    candidates = []
    
    # Enhanced section patterns
    section_patterns = [
        r"(?:Section|Sections|Sec\.?|U/s|U/s\.|धारा|कलम|under\s*section)\s*[:\-]?\s*((?:\d{1,3}[A-Za-z]?(?:\([^)]*\))?(?:\s*,\s*)?)+)",
        r"(?:u/s|under)\s*(\d{1,3}[A-Za-z]?(?:\([^)]*\))?(?:\s*,\s*\d{1,3}[A-Za-z]?(?:\([^)]*\))?)*)",
        r"IPC\s*(\d{1,3}[A-Za-z]?(?:\([^)]*\))?(?:\s*,\s*\d{1,3}[A-Za-z]?(?:\([^)]*\))?)*)",
        r"धारा\s*(\d{1,3}[A-Za-z]?(?:\([^)]*\))?(?:\s*,\s*\d{1,3}[A-Za-z]?(?:\([^)]*\))?)*)",
        r"Section\s*(\d{1,3}[A-Za-z]?(?:\([^)]*\))?(?:\s*[-,&]\s*\d{1,3}[A-Za-z]?(?:\([^)]*\))?)*)"
    ]
    
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            sections_text = match.group(1)
            # Extract individual section numbers
            section_nums = re.findall(r'\d{1,3}[A-Za-z]?(?:\([^)]*\))?', sections_text)
            for section in section_nums:
                # Validate section number
                base_section = re.match(r'(\d{1,3})', section)
                if base_section:
                    try:
                        num = int(base_section.group(1))
                        if 1 <= num <= SECTION_MAX:
                            candidates.append(section.strip())
                    except:
                        continue
    
    # Fallback: look for standalone numbers that might be sections
    if not candidates:
        standalone_numbers = re.findall(r'\b(\d{2,3})\b', text)
        for num in standalone_numbers:
            try:
                val = int(num)
                if 100 <= val <= SECTION_MAX:  # Likely section numbers
                    candidates.append(num)
            except:
                continue
    
    return dedupe_preserve_order(candidates)

def find_complainant_accused_candidates(text: str) -> Dict[str, List[str]]:
    result = {"complainant": [], "accused": []}
    
    # Enhanced name patterns for complainant
    complainant_patterns = [
        r"(?:Complainant|Informant|तक्रारदार|सूचक|Complainant/Informant|Comp\.?)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s.]{3,50})",
        r"(?:Shri|Smt|Ms|Mr|श्री|श्रीमती)\s*([A-Za-z\u0900-\u097F\s.]{3,50})\s*(?:complainant|complaining|तक्रारदार)",
        r"(?:Name|नाम|नाव)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s.]{3,50})",
        r"I,\s*([A-Za-z\u0900-\u097F\s.]{3,50}),?\s*(?:hereby|want to|wish to|complain)"
    ]
    
    # Enhanced name patterns for accused
    accused_patterns = [
        r"(?:Accused|आरोपी|प्रतिवादी|Accused\s*person)\s*[:\-]?\s*([A-Za-z\u0900-\u097F\s.]{3,50})",
        r"(?:against|विरुद्ध)\s*([A-Za-z\u0900-\u097F\s.]{3,50})",
        r"(?:suspect|संदिग्ध|शक की)\s*([A-Za-z\u0900-\u097F\s.]{3,50})"
    ]
    
    # Extract complainant names
    for pattern in complainant_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.group(1).strip()
            name = re.sub(r'\b(?:complainant|informant|name|son|daughter|w/o|s/o|d/o)\b', '', name, flags=re.IGNORECASE).strip()
            if not is_placeholder_text(name) and len(name) >= 3:
                result["complainant"].append(name)
    
    # Extract accused names
    for pattern in accused_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.group(1).strip()
            name = re.sub(r'\b(?:accused|suspect|person|son|daughter|w/o|s/o|d/o)\b', '', name, flags=re.IGNORECASE).strip()
            if not is_placeholder_text(name) and len(name) >= 3:
                result["accused"].append(name)
    
    # General name extraction (proper nouns)
    proper_noun_pattern = r'\b([A-Z][a-z]{2,15}(?:\s+[A-Z][a-z]{2,15}){1,3})\b'
    proper_names = re.findall(proper_noun_pattern, text)
    
    # Context-based classification
    for name in proper_names:
        if not is_placeholder_text(name):
            # Check context around the name
            name_context = ""
            name_pos = text.find(name)
            if name_pos != -1:
                start = max(0, name_pos - 50)
                end = min(len(text), name_pos + len(name) + 50)
                name_context = text[start:end].lower()
            
            if any(word in name_context for word in ["complainant", "informant", "तक्रारदार", "सूचक"]):
                result["complainant"].append(name)
            elif any(word in name_context for word in ["accused", "आरोपी", "against", "विरुद्ध"]):
                result["accused"].append(name)
    
    for key in result:
        result[key] = dedupe_preserve_order(result[key])
    
    return result

def find_address_candidates(text: str) -> List[str]:
    candidates = []
    
    # Enhanced address patterns
    address_patterns = [
        r"(?:Address|पत्ता|पत्ता\s*[:\-]*|निवास|निवासी)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\-\s]{10,200})",
        r"(?:Resident\s*of|R/o|निवासी)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\-\s]{10,200})",
        r"(?:Village|गाव|ग्राम|Ward|वार्ड)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\-\s]{5,200})",
        r"(?:Tehsil|तहसील|Taluka|तालुका)\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F,./\-\s]{3,200})"
    ]
    
    for pattern in address_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            address = match.group(1).strip()
            # Stop at phone/mobile/other fields
            address = re.split(r'(?:Phone|Mobile|Tel|मोबाइल|मोबा|फोन|UID|Passport|Age|उमर)', address, flags=re.IGNORECASE)[0].strip()
            if len(address) >= 10 and not is_placeholder_text(address):
                candidates.append(address)
    
    # Look for addresses with PIN codes
    pin_code_pattern = r'([A-Za-z0-9\u0900-\u097F,./\-\s]{10,200})\s*(?:PIN|Pin|पिन)\s*[:\-]?\s*\d{6}'
    pin_matches = re.finditer(pin_code_pattern, text, re.IGNORECASE)
    for match in pin_matches:
        candidates.append(match.group(1).strip())
    
    # Look for addresses ending with PIN codes
    address_with_pin = r'([A-Za-z0-9\u0900-\u097F,./\-\s]{15,200})\s+\d{6}\b'
    pin_addr_matches = re.finditer(address_with_pin, text)
    for match in pin_addr_matches:
        candidates.append(match.group(1).strip())
    
    return dedupe_preserve_order(candidates)

def determine_case_category(sections: List[str], acts: List[str]) -> str:
    if not sections:
        return "OTHER"
    
    section_set = set(sections)
    
    # Check each category
    for category, category_sections in CASE_CATEGORIES.items():
        if any(sec in section_set for sec in category_sections):
            return category
    
    # Check based on acts
    if acts:
        for act in acts:
            if "Information Technology" in act or "Cyber" in act.lower():
                return "CYBER_CRIME"
            elif "POCSO" in act:
                return "SEXUAL_OFFENCE"
            elif "NDPS" in act:
                return "DRUG_OFFENCE"
            elif "Dowry" in act:
                return "DOWRY"
    
    return "OTHER"

def determine_jurisdiction_info(state: str, district: str) -> Tuple[Optional[str], Optional[str]]:
    if district:
        return district, "DISTRICT"
    elif state:
        return state, "STATE"
    else:
        return None, None

# ---------------- Enhanced NER Support ----------------

def extract_with_ner(text: str, ner_pipe) -> Dict[str, List[str]]:
    result = {"persons": [], "locations": [], "organizations": []}
    
    if ner_pipe is None:
        return result
    
    try:
        # Process in chunks to avoid memory issues
        text_chunk = text[:8000]  # First 8000 characters
        entities = ner_pipe(text_chunk)
        
        for entity in entities:
            entity_group = entity.get("entity_group") or entity.get("entity", "")
            word = entity.get("word", "").strip()
            
            if not word or is_placeholder_text(word):
                continue
            
            # Clean up subword tokens
            word = word.replace("##", "").replace("▁", "")
            
            if entity_group in ["PER", "PERSON"]:
                result["persons"].append(word)
            elif entity_group in ["LOC", "LOCATION", "GPE"]:
                result["locations"].append(word)
            elif entity_group in ["ORG", "ORGANIZATION"]:
                result["organizations"].append(word)
    
    except Exception as e:
        st.warning(f"NER processing failed: {str(e)}")
    
    for key in result:
        result[key] = dedupe_preserve_order(result[key])
    
    return result

# ---------------- Enhanced Main Extraction Function ----------------

def extract_all_fields(text: str, use_ner: bool = True, ner_pipe=None) -> Dict[str, Any]:
    if not text:
        return {}
    
    # Canonicalize text
    clean_text = canonicalize_text(text)
    
    # Extract all candidate fields
    fir_candidates = find_fir_number_candidates(clean_text)
    year_candidates = find_year_candidates(clean_text)
    state_candidates = find_state_candidates(clean_text)
    district_candidates = find_district_candidates(clean_text)
    ps_candidates = find_police_station_candidates(clean_text)
    acts_candidates = find_acts_candidates(clean_text)
    sections_candidates = find_sections_candidates(clean_text)
    name_data = find_complainant_accused_candidates(clean_text)
    address_candidates = find_address_candidates(clean_text)
    
    # NER enhancement
    ner_data = {"persons": [], "locations": [], "organizations": []}
    if use_ner and ner_pipe is not None:
        ner_data = extract_with_ner(clean_text, ner_pipe)
    
    # Select best candidates
    fir_no = fir_candidates[0] if fir_candidates else None
    year = year_candidates[0] if year_candidates else None
    state = state_candidates[0] if state_candidates else None
    district = district_candidates[0] if district_candidates else None
    police_station = ps_candidates[0] if ps_candidates else None
    
    # Combine complainant names from regex and NER
    all_complainants = name_data["complainant"] + ner_data["persons"]
    complainant_name = all_complainants[0] if all_complainants else None
    
    # Combine accused names
    all_accused = name_data["accused"]
    accused_name = all_accused[0] if all_accused else None
    
    # Determine primary party type
    oparty = None
    if complainant_name and not accused_name:
        oparty = "Complainant"
    elif accused_name and not complainant_name:
        oparty = "Accused"
    elif complainant_name:
        oparty = "Complainant"
    
    # Primary name (complainant takes priority)
    primary_name = complainant_name or accused_name
    
    # Address
    all_addresses = address_candidates + ner_data["locations"]
    address = all_addresses[0] if all_addresses else None
    
    # Case categorization
    case_category = determine_case_category(sections_candidates, acts_candidates)
    
    # Jurisdiction
    jurisdiction, jurisdiction_type = determine_jurisdiction_info(state, district)
    
    # Build result
    result = {
        "fir_no": clean_and_validate_field(fir_no, 50),
        "year": clean_and_validate_field(year, 4),
        "state_name": clean_and_validate_field(state, 100),
        "dist_name": clean_and_validate_field(district, 100),
        "police_station": clean_and_validate_field(police_station, 100),
        "under_acts": acts_candidates[:5] if acts_candidates else None,  # Limit to 5
        "under_sections": sections_candidates[:10] if sections_candidates else None,  # Limit to 10
        "revised_case_category": case_category,
        "oparty": oparty,
        "name": clean_and_validate_field(primary_name, 200),
        "address": clean_and_validate_field(address, 500),
        "jurisdiction": clean_and_validate_field(jurisdiction, 100),
        "jurisdiction_type": jurisdiction_type
    }
    
    return result

# ---------------- Streamlit UI ----------------

st.title("🚔 Enhanced FIR PII Extractor")
st.write("Advanced extraction with improved patterns, NER support, and better field recognition.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_ner = st.checkbox("Use NER Enhancement", value=TRANSFORMERS_AVAILABLE, 
                         help="Uses AI models to identify names and locations")
    tesseract_langs = st.text_input("Tesseract Languages", value="eng+hin+mar",
                                   help="Language codes for OCR (install language packs)")
    
    st.markdown("---")
    st.subheader("📋 Extracted Fields")
    st.markdown("""
    - **FIR Number**: Registration number
    - **Year**: Year of registration  
    - **State/District**: Location details
    - **Police Station**: Reporting station
    - **Acts & Sections**: Legal provisions
    - **Category**: Auto-classified case type
    - **Names**: Complainant/Accused
    - **Address**: Location information
    - **Jurisdiction**: Administrative scope
    """)

# File upload
uploaded_files = st.file_uploader(
    "📁 Upload FIR PDFs", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Upload one or more FIR PDF files"
)

# Text input
pasted_text = st.text_area(
    "📝 Or paste FIR text directly", 
    height=300,
    placeholder="Paste the FIR content here..."
)

# Load NER pipeline
ner_pipeline = None
if use_ner:
    with st.spinner("Loading NER model..."):
        ner_pipeline = load_ner_pipe()
        if ner_pipeline is None and TRANSFORMERS_AVAILABLE:
            st.warning("⚠️ NER model failed to load. Proceeding with regex-only extraction.")

# Process button
if st.button("🔍 Extract Information", type="primary"):
    results = {}
    
    # Process uploaded PDFs
    if uploaded_files:
        progress_bar = st.progress(0)
        for i, uploaded_file in enumerate(uploaded_files):
            st.info(f"Processing: {uploaded_file.name}")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Extract text from PDF
                extracted_text = extract_text_from_pdf(tmp_path, tesseract_langs)
                
                if not extracted_text.strip():
                    st.warning(f"❌ No text could be extracted from {uploaded_file.name}")
                    continue
                
                # Extract fields
                result = extract_all_fields(
                    extracted_text, 
                    use_ner=use_ner, 
                    ner_pipe=ner_pipeline
                )
                results[uploaded_file.name] = result
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                try:
                    os.remove(tmp_path)
                except:
                    pass
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
    
    # Process pasted text
    if pasted_text and pasted_text.strip():
        st.info("Processing pasted text...")
        try:
            result = extract_all_fields(
                pasted_text, 
                use_ner=use_ner, 
                ner_pipe=ner_pipeline
            )
            results["pasted_text"] = result
        except Exception as e:
            st.error(f"❌ Error processing pasted text: {str(e)}")
    
    # Display results
    if not results:
        st.warning("⚠️ Please upload PDF files or paste FIR text to extract information.")
    else:
        st.success(f"✅ Successfully processed {len(results)} item(s)")
        
        # Display results in tabs
        if len(results) == 1:
            # Single result - display directly
            result = list(results.values())[0]
            st.subheader("📊 Extracted Information")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Case Details**")
                st.write(f"**FIR Number:** {result.get('fir_no', 'Not found')}")
                st.write(f"**Year:** {result.get('year', 'Not found')}")
                st.write(f"**Category:** {result.get('revised_case_category', 'OTHER')}")
                
                st.markdown("**🏛️ Location**")
                st.write(f"**State:** {result.get('state_name', 'Not found')}")
                st.write(f"**District:** {result.get('dist_name', 'Not found')}")
                st.write(f"**Police Station:** {result.get('police_station', 'Not found')}")
                
            with col2:
                st.markdown("**👤 Parties**")
                st.write(f"**Primary Party:** {result.get('oparty', 'Not determined')}")
                st.write(f"**Name:** {result.get('name', 'Not found')}")
                
                st.markdown("**⚖️ Legal Provisions**")
                acts = result.get('under_acts', [])
                sections = result.get('under_sections', [])
                st.write(f"**Acts:** {', '.join(acts) if acts else 'Not found'}")
                st.write(f"**Sections:** {', '.join(sections) if sections else 'Not found'}")
            
            # Address in full width
            if result.get('address'):
                st.markdown("**📍 Address**")
                st.text_area("", value=result['address'], height=100, disabled=True)
            
            # Jurisdiction info
            if result.get('jurisdiction'):
                st.markdown("**🗺️ Jurisdiction**")
                st.write(f"{result['jurisdiction']} ({result.get('jurisdiction_type', 'Unknown')})")
            
        else:
            # Multiple results - use tabs
            tabs = st.tabs([name for name in results.keys()])
            for tab, (name, result) in zip(tabs, results.items()):
                with tab:
                    st.json(result, expanded=True)
        
        # Download option
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        json_data = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"fir_extraction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Show raw JSON for debugging
        with st.expander("🔍 View Raw JSON"):
            st.json(results, expanded=False)

# Help section
with st.expander("❓ Help & Tips"):
    st.markdown("""
    ### 🎯 How to get better results:
    
    **For PDFs:**
    - Upload clear, high-quality scanned PDFs
    - Ensure text is readable (not too blurry or distorted)
    - For Hindi text, make sure appropriate language packs are installed
    
    **For pasted text:**
    - Include complete FIR information
    - Maintain original formatting when possible
    - Include section numbers, acts, and location details
    
    ### 🔧 Features:
    - **Multi-language support**: Hindi, Marathi, English
    - **Smart categorization**: Auto-classifies case types
    - **Fuzzy matching**: Corrects common spelling variations
    - **NER enhancement**: AI-powered name/location detection
    - **Multiple extraction methods**: PDF parsing + OCR fallback
    
    ### 📋 Extracted Fields:
    - FIR Number, Year, Location details
    - Acts and Sections under which case is registered
    - Complainant/Accused information
    - Case category classification
    - Jurisdiction information
    """)
