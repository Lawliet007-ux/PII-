import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime
from typing import Dict, List, Set, Any, Tuple
import spacy
from spacy import displacy
import warnings
warnings.filterwarnings('ignore')

# For text cleaning and preprocessing
import unicodedata
from collections import Counter

# For Elasticsearch connection
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, AuthenticationException

# For advanced NER models
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# For phone number detection
import phonenumbers
from phonenumbers import geocoder, carrier

# For email validation
import email_validator

# Configuration
st.set_page_config(
    page_title="OCR-Enhanced PII Extraction Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OCRTextCleaner:
    """Specialized class for cleaning OCR-processed text"""
    
    def __init__(self):
        # Common OCR character replacements
        self.ocr_replacements = {
            # Common OCR errors
            '⁇': '',
            'ã¡a': 'ा',
            'Đ': 'ड',
            'Í': 'ी',
            'Ö': 'ो',
            '¡': 'ि',
            '⇢': '→',
            'è': 'े',
            'ğ': 'ग',
            'ç': 'च',
            
            # Common symbol errors
            '\\': '',
            '\\.': '.',
            '\\n': ' ',
            '\\\\': ' ',
            '\\ ': ' ',
            
            # Multiple spaces
            r'\s+': ' ',
            
            # Remove unknown Unicode characters
            r'[^\w\s\u0900-\u097F\u0020-\u007E\.\,\-\:\(\)\[\]\/\@]': '',
        }
        
        # Contextual patterns for better extraction
        self.context_patterns = {
            'fir_no': r'(?i)(?:fir\s*(?:no|number)?\.?\s*[:()]?\s*)([0-9]{3,6})',
            'police_station': r'(?i)(?:p\.?s\.?|police\s+(?:station|thane))[:\s]*([a-zA-Z\s\u0900-\u097F]+)',
            'district': r'(?i)(?:district)[:\s]*([a-zA-Z\s\u0900-\u097F]+)',
            'date_time': r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s*(?:at\s*)?(\d{1,2}[:]\d{2})',
            'address_context': r'(?i)(?:address|पता)[:\s]*([a-zA-Z0-9\s\u0900-\u097F\,\.\-]+?)(?=\n|\d\.|\w+:)',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean OCR artifacts from text"""
        if not text:
            return ""
        
        cleaned = text
        
        # Apply OCR character replacements
        for ocr_char, replacement in self.ocr_replacements.items():
            if r'\s+' in ocr_char:
                cleaned = re.sub(ocr_char, replacement, cleaned)
            else:
                cleaned = cleaned.replace(ocr_char, replacement)
        
        # Normalize Unicode characters
        cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Clean up common OCR patterns
        cleaned = re.sub(r'[\\]{2,}', ' ', cleaned)
        cleaned = re.sub(r'[\.]{3,}', '...', cleaned)
        cleaned = re.sub(r'[:]{2,}', ':', cleaned)
        
        return cleaned.strip()
    
    def extract_contextual_info(self, text: str) -> Dict[str, List[str]]:
        """Extract information using contextual patterns"""
        results = {}
        cleaned_text = self.clean_text(text)
        
        for info_type, pattern in self.context_patterns.items():
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Handle different match types
                if info_type == 'date_time':
                    # Combine date and time matches
                    formatted_matches = []
                    for match in matches:
                        if isinstance(match, tuple) and len(match) == 2:
                            formatted_matches.append(f"{match[0]} {match[1]}")
                        else:
                            formatted_matches.append(str(match))
                    results[info_type] = formatted_matches
                else:
                    # Clean and filter matches
                    cleaned_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match else ""
                        
                        match = str(match).strip()
                        if len(match) > 2 and not match.isdigit():
                            cleaned_matches.append(match)
                    
                    if cleaned_matches:
                        results[info_type] = cleaned_matches
        
        return results

class EnhancedPIIExtractor:
    def __init__(self):
        self.text_cleaner = OCRTextCleaner()
        self.pii_patterns = self._load_enhanced_patterns()
        self.hindi_ner_model = None
        self.english_ner_model = None
        self.multilingual_ner = None
        
    def _load_enhanced_patterns(self) -> Dict[str, str]:
        """Load enhanced regex patterns for OCR text"""
        return {
            # Enhanced Indian specific patterns (OCR-tolerant)
            'aadhaar': r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
            'aadhaar_verbose': r'(?i)(?:aadhaar|aadhar|आधार)[\s\w]*?[:]\s*(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})',
            
            'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            'pan_verbose': r'(?i)(?:pan|पैन)[\s\w]*?[:]\s*([A-Z]{5}[0-9]{4}[A-Z]{1})',
            
            # More flexible phone patterns
            'indian_mobile': r'\b(?:\+91[\s\-]?|0)?[6-9]\d{9}\b',
            'mobile_verbose': r'(?i)(?:mobile|phone|mob|फोन|मोबाइल)[\s\w]*?[:]\s*((?:\+91[\s\-]?|0)?[6-9]\d{9})',
            
            # Legal document specific
            'fir_number': r'(?i)(?:fir|एफआईआर)[\s\w]*?[:]\s*([0-9]{3,6})',
            'case_number': r'(?i)(?:case|केस)[\s\w]*?[:]\s*([A-Z0-9\/\-]{4,20})',
            
            # Enhanced address patterns
            'pincode': r'\b[1-9]\d{5}\b',
            'pincode_verbose': r'(?i)(?:pin|pincode|पिन)[\s\w]*?[:]\s*([1-9]\d{5})',
            
            # Document identifiers
            'gstin': r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b',
            'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            'indian_passport': r'\b[A-Z]{1}\d{7}\b',
            'voter_id': r'\b[A-Z]{3}\d{7}\b',
            'driving_license': r'\b[A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{7}\b',
            
            # Enhanced email pattern (OCR tolerant)
            'email': r'\b[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'email_ocr': r'\b[A-Za-z0-9._%-]+[@]\s*[A-Za-z0-9.-]+[.]\s*[A-Z|a-z]{2,}\b',
            
            # Date patterns
            'date': r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
            'time': r'\b\d{1,2}[:]\d{2}(?:\s*(?:hours|hrs))?\b',
            
            # Name patterns (enhanced for OCR)
            'names_context': r'(?i)(?:name|नाम|accused|आरोपी|complainant)[\s\w]*?[:]\s*([A-Za-z\s\u0900-\u097F]{3,50})',
            
            # Address components
            'address_keywords': r'(?i)\b(?:house|flat|apartment|block|sector|phase|colony|nagar|road|street|lane|gali|marg|chowk|circle|square|pune|mumbai|delhi|bangalore|hyderabad|chennai|kolkata)\b',
        }
    
    @st.cache_resource
    def load_models(_self):
        """Load NER models with enhanced error handling"""
        try:
            # Try loading better multilingual models
            model_options = [
                "dbmdz/bert-large-cased-finetuned-conll03-english",
                "xlm-roberta-large-finetuned-conll03-english", 
                "microsoft/DialoGPT-medium"
            ]
            
            for model_name in model_options:
                try:
                    _self.multilingual_ner = pipeline(
                        "ner", 
                        model=model_name,
                        tokenizer=model_name,
                        aggregation_strategy="simple"
                    )
                    st.success(f"Loaded NER model: {model_name}")
                    break
                except Exception as e:
                    st.warning(f"Failed to load {model_name}: {str(e)[:100]}")
                    continue
            
            # Load spaCy model for English
            try:
                import en_core_web_sm
                _self.english_nlp = en_core_web_sm.load()
            except:
                try:
                    import spacy
                    _self.english_nlp = spacy.blank("en")
                    st.warning("Using blank English model - install en_core_web_sm for better results")
                except:
                    _self.english_nlp = None
                    
        except Exception as e:
            st.error(f"Error loading NER models: {e}")
            _self.multilingual_ner = None
    
    def extract_enhanced_regex_pii(self, text: str) -> Dict[str, List[str]]:
        """Extract PII using enhanced OCR-tolerant patterns"""
        # Clean the text first
        cleaned_text = self.text_cleaner.clean_text(text)
        extracted_pii = {}
        
        # Extract contextual information first
        contextual_info = self.text_cleaner.extract_contextual_info(text)
        extracted_pii.update(contextual_info)
        
        # Apply regex patterns
        for pii_type, pattern in self.pii_patterns.items():
            try:
                matches = re.findall(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Handle tuple matches (from groups)
                    processed_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            # Take the first non-empty group
                            match_str = next((m for m in match if m), '')
                        else:
                            match_str = str(match)
                        
                        match_str = match_str.strip()
                        if len(match_str) > 1:
                            processed_matches.append(match_str)
                    
                    # Remove duplicates while preserving order
                    unique_matches = list(dict.fromkeys(processed_matches))
                    if unique_matches:
                        extracted_pii[pii_type] = unique_matches
            except Exception as e:
                st.warning(f"Pattern matching failed for {pii_type}: {str(e)[:50]}")
        
        return extracted_pii
    
    def extract_names_advanced(self, text: str) -> List[str]:
        """Advanced name extraction for OCR text"""
        names = []
        cleaned_text = self.text_cleaner.clean_text(text)
        
        # Pattern-based name extraction
        name_patterns = [
            r'(?i)(?:name|नाम)[\s\w]*?[:]\s*([A-Za-z\s\u0900-\u097F]{3,40})',
            r'(?i)(?:accused|आरोपी)[\s\w]*?[:]\s*([A-Za-z\s\u0900-\u097F]{3,40})',
            r'(?i)(?:complainant|शिकायतकर्ता)[\s\w]*?[:]\s*([A-Za-z\s\u0900-\u097F]{3,40})',
            r'(?i)(?:witness|गवाह)[\s\w]*?[:]\s*([A-Za-z\s\u0900-\u097F]{3,40})',
            r'(?i)(?:father|पिता|son|पुत्र|daughter|पुत्री)[\s\w]*?[:]\s*([A-Za-z\s\u0900-\u097F]{3,40})',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, cleaned_text, re.MULTILINE)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 2:
                    # Clean the name
                    name = re.sub(r'\s+', ' ', match.strip())
                    # Remove common false positives
                    if not re.match(r'^\d+$', name) and name.lower() not in ['date', 'time', 'case', 'fir']:
                        names.append(name)
        
        return list(dict.fromkeys(names))  # Remove duplicates
    
    def extract_addresses_advanced(self, text: str) -> List[str]:
        """Advanced address extraction for OCR text"""
        addresses = []
        cleaned_text = self.text_cleaner.clean_text(text)
        
        # Address patterns with context
        address_patterns = [
            r'(?i)(?:address|पता|residence|निवास)[\s\w]*?[:]\s*([A-Za-z0-9\s\u0900-\u097F\,\.\-\/]{10,200}?)(?=\n|\d\.|[A-Z][a-z]+:)',
            r'(?i)(?:at|में)\s+([A-Za-z\s\u0900-\u097F]{3,50}(?:nagar|colony|road|street|marg|pune|mumbai|delhi)[A-Za-z0-9\s\u0900-\u097F\,\.\-]{0,100})',
        ]
        
        for pattern in address_patterns:
            matches = re.findall(pattern, cleaned_text, re.MULTILINE)
            for match in matches:
                if isinstance(match, str) and len(match.strip()) > 5:
                    # Clean the address
                    address = re.sub(r'\s+', ' ', match.strip())
                    # Basic validation
                    if len(address.split()) >= 2:
                        addresses.append(address[:200])  # Limit length
        
        return list(dict.fromkeys(addresses))
    
    def is_likely_pii_enhanced(self, text: str, pii_type: str) -> bool:
        """Enhanced validation with OCR considerations"""
        text = text.strip()
        
        # Skip very short matches
        if len(text) < 2:
            return False
        
        # Skip common OCR artifacts
        ocr_artifacts = ['⁇', 'ã¡a', 'Đ', '\\', '...', ':::', '---']
        if any(artifact in text for artifact in ocr_artifacts):
            return False
            
        # Skip common words
        common_words = {
            'english': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'date', 'time', 'case', 'fir', 'section'],
            'hindi': ['और', 'या', 'में', 'पर', 'से', 'के', 'का', 'की', 'है', 'हैं', 'था', 'थी', 'दिनांक', 'समय']
        }
        
        if text.lower() in common_words['english'] or text in common_words['hindi']:
            return False
        
        # Type-specific enhanced validation
        if pii_type == 'aadhaar' or pii_type == 'aadhaar_verbose':
            clean = re.sub(r'[\s\-]', '', text)
            return len(clean) == 12 and clean.isdigit() and not clean.startswith('0')
            
        elif pii_type == 'pan' or pii_type == 'pan_verbose':
            clean = text.replace(' ', '').replace('-', '').upper()
            return len(clean) == 10 and re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', clean)
            
        elif pii_type in ['indian_mobile', 'mobile_verbose']:
            clean = re.sub(r'[\s\-\+]', '', text)
            if clean.startswith('91'):
                clean = clean[2:]
            elif clean.startswith('0'):
                clean = clean[1:]
            return len(clean) == 10 and clean.startswith(('6', '7', '8', '9'))
            
        elif pii_type in ['pincode', 'pincode_verbose']:
            return len(text) == 6 and text.isdigit() and not text.startswith('0')
            
        elif pii_type in ['email', 'email_ocr']:
            # Basic email validation
            return '@' in text and '.' in text.split('@')[-1] and len(text) > 5
            
        elif 'names' in pii_type:
            # Name validation
            return len(text.split()) <= 6 and not text.isdigit() and len(text) >= 3
            
        elif 'address' in pii_type:
            # Address validation
            return len(text.split()) >= 2 and len(text) >= 10
            
        elif pii_type in ['fir_number', 'case_number']:
            # Legal document numbers
            return len(text) >= 3 and any(c.isdigit() for c in text)
        
        return True
    
    def extract_all_pii(self, text: str) -> Dict[str, Any]:
        """Main function to extract all PII from OCR text"""
        if not text or len(text.strip()) < 10:
            return {}
        
        # Extract using enhanced regex patterns
        regex_pii = self.extract_enhanced_regex_pii(text)
        
        # Extract names using advanced methods
        extracted_names = self.extract_names_advanced(text)
        if extracted_names:
            regex_pii['extracted_names'] = extracted_names
        
        # Extract addresses using advanced methods
        extracted_addresses = self.extract_addresses_advanced(text)
        if extracted_addresses:
            regex_pii['extracted_addresses'] = extracted_addresses
        
        # Extract using NER models (if available)
        if hasattr(self, 'english_nlp') and self.english_nlp:
            try:
                cleaned_text = self.text_cleaner.clean_text(text)
                doc = self.english_nlp(cleaned_text)
                
                ner_persons = []
                ner_orgs = []
                ner_locations = []
                
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
                        ner_persons.append(ent.text.strip())
                    elif ent.label_ in ["ORG", "ORGANIZATION"] and len(ent.text.strip()) > 2:
                        ner_orgs.append(ent.text.strip())
                    elif ent.label_ in ["GPE", "LOC", "LOCATION"] and len(ent.text.strip()) > 2:
                        ner_locations.append(ent.text.strip())
                
                if ner_persons:
                    regex_pii['ner_persons'] = list(dict.fromkeys(ner_persons))
                if ner_orgs:
                    regex_pii['ner_organizations'] = list(dict.fromkeys(ner_orgs))
                if ner_locations:
                    regex_pii['ner_locations'] = list(dict.fromkeys(ner_locations))
                    
            except Exception as e:
                st.warning(f"NER extraction failed: {str(e)[:50]}")
        
        # Validate and filter results
        final_pii = {}
        for pii_type, matches in regex_pii.items():
            if matches:
                validated_matches = []
                for match in matches:
                    if self.is_likely_pii_enhanced(str(match), pii_type):
                        validated_matches.append(match)
                
                if validated_matches:
                    # Remove duplicates while preserving order
                    final_pii[pii_type] = list(dict.fromkeys(validated_matches))
        
        # Special validation for phones and emails
        if 'indian_mobile' in final_pii or 'mobile_verbose' in final_pii:
            all_phones = final_pii.get('indian_mobile', []) + final_pii.get('mobile_verbose', [])
            if all_phones:
                final_pii['validated_phones'] = self.validate_phone_numbers(all_phones)
        
        if 'email' in final_pii or 'email_ocr' in final_pii:
            all_emails = final_pii.get('email', []) + final_pii.get('email_ocr', [])
            if all_emails:
                final_pii['validated_emails'] = self.validate_emails(all_emails)
        
        return final_pii
    
    def validate_phone_numbers(self, phone_numbers: List[str]) -> List[Dict]:
        """Validate phone numbers with OCR tolerance"""
        validated_phones = []
        
        for phone in phone_numbers:
            try:
                # Clean the phone number more aggressively
                clean_phone = re.sub(r'[^\d+]', '', str(phone))
                
                # Handle Indian numbers
                if not clean_phone.startswith('+'):
                    if clean_phone.startswith('91') and len(clean_phone) > 10:
                        clean_phone = '+91' + clean_phone[2:]
                    elif clean_phone.startswith('0'):
                        clean_phone = '+91' + clean_phone[1:]
                    elif len(clean_phone) == 10:
                        clean_phone = '+91' + clean_phone
                
                parsed = phonenumbers.parse(clean_phone, "IN")
                
                if phonenumbers.is_valid_number(parsed):
                    validated_phones.append({
                        'original': phone,
                        'formatted': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                        'country': geocoder.description_for_number(parsed, "en"),
                        'carrier': carrier.name_for_number(parsed, "en"),
                        'valid': True
                    })
                else:
                    # Include as potential PII even if validation fails
                    validated_phones.append({
                        'original': phone,
                        'cleaned': clean_phone,
                        'valid': False
                    })
                    
            except Exception as e:
                validated_phones.append({
                    'original': phone,
                    'error': str(e)[:50],
                    'valid': False
                })
        
        return validated_phones
    
    def validate_emails(self, emails: List[str]) -> List[Dict]:
        """Validate emails with OCR tolerance"""
        validated_emails = []
        
        for email in emails:
            try:
                # Clean common OCR errors in emails
                clean_email = str(email).strip()
                clean_email = re.sub(r'\s+', '', clean_email)  # Remove spaces
                
                valid_email = email_validator.validate_email(clean_email)
                validated_emails.append({
                    'original': email,
                    'normalized': valid_email.email,
                    'valid': True
                })
            except Exception as e:
                validated_emails.append({
                    'original': email,
                    'error': str(e)[:50],
                    'valid': False
                })
        
        return validated_emails

class ElasticsearchConnector:
    def __init__(self, host: str, port: int = 9200, username: str = None, password: str = None, use_ssl: bool = False):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.es = None
        
    def connect(self) -> bool:
        """Connect to Elasticsearch"""
        try:
            if self.username and self.password:
                self.es = Elasticsearch(
                    [{'host': self.host, 'port': self.port}],
                    http_auth=(self.username, self.password),
                    use_ssl=self.use_ssl,
                    verify_certs=False,
                    timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
            else:
                self.es = Elasticsearch(
                    [{'host': self.host, 'port': self.port}],
                    use_ssl=self.use_ssl,
                    verify_certs=False,
                    timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
            
            # Test connection
            if self.es.ping():
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Connection failed: {e}")
            return False
    
    def get_indices(self) -> List[str]:
        """Get list of available indices"""
        try:
            if self.es:
                indices = self.es.indices.get_alias().keys()
                return sorted([idx for idx in indices if not idx.startswith('.')])
        except Exception as e:
            st.error(f"Error fetching indices: {e}")
        return []
    
    def search_data(self, index: str, query: Dict = None, size: int = 100) -> List[Dict]:
        """Search data from Elasticsearch"""
        try:
            if not query:
                query = {"match_all": {}}
            
            response = self.es.search(
                index=index,
                body={"query": query, "size": size}
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

def main():
    st.title("🔍 OCR-Enhanced PII Extraction Tool")
    st.markdown("**Specialized for OCR-processed documents (Police FIRs, Legal Documents, etc.)**")
    
    # Initialize session state
    if 'pii_extractor' not in st.session_state:
        st.session_state.pii_extractor = EnhancedPIIExtractor()
        
    if 'es_connector' not in st.session_state:
        st.session_state.es_connector = None
        
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Elasticsearch configuration
        st.subheader("Elasticsearch Connection")
        es_host = st.text_input("Host", value="localhost", help="Elasticsearch host")
        es_port = st.number_input("Port", value=9200, help="Elasticsearch port")
        es_username = st.text_input("Username (optional)", help="Leave empty if no auth")
        es_password = st.text_input("Password (optional)", type="password")
        use_ssl = st.checkbox("Use SSL", help="Enable SSL connection")
        
        if st.button("Connect to Elasticsearch"):
            with st.spinner("Connecting..."):
                connector = ElasticsearchConnector(es_host, es_port, es_username, es_password, use_ssl)
                if connector.connect():
                    st.session_state.es_connector = connector
                    st.success("✅ Connected successfully!")
                else:
                    st.error("❌ Connection failed!")
        
        # Model loading
        st.subheader("Load NER Models")
        if st.button("Load Models"):
            with st.spinner("Loading models... This may take a few minutes"):
                st.session_state.pii_extractor.load_models()
                st.success("✅ Models loaded!")
        
        # OCR Tips
        st.subheader("💡 OCR Processing Tips")
        st.info("""
        This tool is optimized for:
        - Police FIR documents
        - Legal case files
        - Scanned/OCR'd documents
        - Mixed Hindi/English text
        - Documents with OCR artifacts
        """)
    
    # Show example of OCR text processing
    with st.expander("🔍 See OCR Text Cleaning Example"):
        example_ocr = """P.S. (Police Thane): Bhosari \\nFIR No. ( ⁇ M Khab Đ.): 0523 \\nDate and Time of FIR ( ⁇ . True. Dated time): \\n19/11/2017 at 21:33 \\nDistrict (Jesus ã¡a): Pune City"""
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original OCR Text:**")
            st.text(example_ocr)
            
        with col2:
            st.write("**Cleaned Text:**")
            cleaned_example = st.session_state.pii_extractor.text_cleaner.clean_text(example_ocr)
            st.text(cleaned_example)
            
            if st.button("Test Example"):
                pii_demo = st.session_state.pii_extractor.extract_all_pii(example_ocr)
                st.json(pii_demo)
    
    # Main interface
    if st.session_state.es_connector and st.session_state.es_connector.es:
        
        # Index selection
        st.subheader("📊 Select Data Source")
        indices = st.session_state.es_connector.get_indices()
        
        if indices:
            selected_index = st.selectbox("Choose Index", indices)
            
            # Query configuration
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                query_type = st.selectbox(
                    "Query Type",
                    ["Match All", "Custom Query", "Text Search", "Date Range"]
                )
                
            with col2:
                max_docs = st.number_input("Max Documents", min_value=1, max_value=1000, value=50)
                
            with col3:
                confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, help="Higher = fewer false positives")
            
            # Build query based on type
            query = {"match_all": {}}
            
            if query_type == "Text Search":
                search_text = st.text_input("Search Text", placeholder="Enter keywords to search")
                if search_text:
                    query = {
                        "multi_match": {
                            "query": search_text,
                            "fields": ["*"],
                            "fuzziness": "AUTO"
                        }
                    }
                    
            elif query_type == "Date Range":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date")
                with col2:
                    end_date = st.date_input("End Date")
                    
                date_field = st.text_input("Date Field Name", value="@timestamp", help="Field containing date")
                
                if start_date and end_date:
                    query = {
                        "range": {
                            date_field: {
                                "gte": start_date.isoformat(),
                                "lte": end_date.isoformat()
                            }
                        }
                    }
                    
            elif query_type == "Custom Query":
                custom_query = st.text_area(
                    "Custom Elasticsearch Query (JSON)",
                    value='{"match_all": {}}',
                    height=150,
                    help="Enter valid Elasticsearch query JSON"
                )
                try:
                    query = json.loads(custom_query)
                except json.JSONDecodeError:
                    st.error("Invalid JSON query")
                    query = {"match_all": {}}
            
            # PII Type selection
            st.subheader("🎯 PII Types to Extract")
            pii_categories = {
                "Identity Documents": ['aadhaar', 'pan', 'voter_id', 'indian_passport', 'driving_license'],
                "Contact Information": ['indian_mobile', 'email', 'address'],
                "Financial": ['gstin', 'ifsc', 'bank_account', 'credit_card'],
                "Personal Info": ['names', 'date_birth', 'pincode'],
                "Legal Documents": ['fir_number', 'case_number']
            }
            
            selected_categories = st.multiselect(
                "Select PII Categories",
                list(pii_categories.keys()),
                default=list(pii_categories.keys())
            )
            
            # Extract PII
            if st.button("🔍 Extract PII", type="primary", help="Start PII extraction process"):
                with st.spinner("Extracting PII from Elasticsearch data..."):
                    
                    # Fetch data
                    documents = st.session_state.es_connector.search_data(selected_index, query, max_docs)
                    
                    if not documents:
                        st.warning("No documents found with the given query.")
                        return
                    
                    st.info(f"Processing {len(documents)} documents...")
                    
                    # Process documents
                    all_pii_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, doc in enumerate(documents):
                        status_text.text(f"Processing document {idx+1}/{len(documents)}")
                        
                        # Convert document to text more intelligently
                        doc_text = ""
                        text_fields = []
                        
                        if isinstance(doc, dict):
                            # Prioritize text-heavy fields
                            priority_fields = ['message', 'content', 'text', 'body', 'description']
                            
                            # Extract from priority fields first
                            for field in priority_fields:
                                if field in doc and isinstance(doc[field], str):
                                    text_fields.append(doc[field])
                            
                            # Extract from all other string fields
                            for key, value in doc.items():
                                if key not in priority_fields and isinstance(value, str) and len(value.strip()) > 5:
                                    text_fields.append(value)
                                elif isinstance(value, list):
                                    for item in value:
                                        if isinstance(item, str) and len(item.strip()) > 5:
                                            text_fields.append(item)
                        else:
                            text_fields.append(str(doc))
                        
                        doc_text = " ".join(text_fields)
                        
                        # Skip very short documents
                        if len(doc_text.strip()) < 20:
                            continue
                        
                        # Extract PII
                        try:
                            pii_found = st.session_state.pii_extractor.extract_all_pii(doc_text)
                            
                            # Filter based on selected categories
                            filtered_pii = {}
                            for category in selected_categories:
                                for pii_type in pii_categories[category]:
                                    if pii_type in pii_found:
                                        filtered_pii[pii_type] = pii_found[pii_type]
                                    # Also check for verbose versions
                                    verbose_type = f"{pii_type}_verbose"
                                    if verbose_type in pii_found:
                                        filtered_pii[verbose_type] = pii_found[verbose_type]
                            
                            # Include special validated types
                            for special_type in ['validated_phones', 'validated_emails', 'extracted_names', 'extracted_addresses', 'ner_persons', 'ner_organizations', 'ner_locations']:
                                if special_type in pii_found:
                                    filtered_pii[special_type] = pii_found[special_type]
                            
                            if filtered_pii:  # Only store documents with relevant PII
                                # Calculate confidence score
                                total_pii_count = sum(len(v) if isinstance(v, list) else 1 for v in filtered_pii.values())
                                confidence_score = min(1.0, total_pii_count / 10.0)  # Normalize to 0-1
                                
                                if confidence_score >= confidence_threshold:
                                    all_pii_results.append({
                                        'document_id': idx,
                                        'document_preview': doc_text[:300] + "..." if len(doc_text) > 300 else doc_text,
                                        'pii_found': filtered_pii,
                                        'confidence_score': confidence_score,
                                        'pii_count': total_pii_count,
                                        'full_document': doc,
                                        'original_text_length': len(doc_text)
                                    })
                        except Exception as e:
                            st.warning(f"Error processing document {idx}: {str(e)[:100]}")
                        
                        progress_bar.progress((idx + 1) / len(documents))
                    
                    st.session_state.extracted_data = all_pii_results
                    progress_bar.empty()
                    status_text.empty()
                    
                    if all_pii_results:
                        st.success(f"✅ Processing complete! Found PII in {len(all_pii_results)} documents.")
                    else:
                        st.warning("⚠️ No PII found in any documents. Try lowering the confidence threshold or check your data.")
        
        else:
            st.info("No indices found in Elasticsearch or connection not established.")
    
    else:
        # Connection instructions
        st.info("👆 Please configure and connect to Elasticsearch in the sidebar to begin.")
        
        st.subheader("🚀 Quick Start Guide")
        st.markdown("""
        1. **Configure Elasticsearch** in the sidebar (host, port, credentials)
        2. **Connect** to your Elasticsearch cluster
        3. **Load Models** for better NER performance
        4. **Select Index** and configure your search query
        5. **Choose PII Categories** you want to extract
        6. **Extract PII** and review results
        """)
    
    # Display results with enhanced visualization
    if st.session_state.extracted_data:
        st.subheader("🎯 PII Extraction Results")
        
        total_docs_with_pii = len(st.session_state.extracted_data)
        total_pii_items = sum(result['pii_count'] for result in st.session_state.extracted_data)
        avg_confidence = sum(result['confidence_score'] for result in st.session_state.extracted_data) / len(st.session_state.extracted_data)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents with PII", total_docs_with_pii)
        col2.metric("Total PII Items", total_pii_items)
        col3.metric("Average Confidence", f"{avg_confidence:.2f}")
        col4.metric("Processing Status", "Complete ✅")
        
        # Enhanced summary statistics
        pii_summary = {}
        confidence_by_type = {}
        
        for result in st.session_state.extracted_data:
            for pii_type, values in result['pii_found'].items():
                if pii_type not in pii_summary:
                    pii_summary[pii_type] = 0
                    confidence_by_type[pii_type] = []
                
                count = len(values) if isinstance(values, list) else 1
                pii_summary[pii_type] += count
                confidence_by_type[pii_type].append(result['confidence_score'])
        
        if pii_summary:
            st.subheader("📈 PII Summary Dashboard")
            
            # Create enhanced summary DataFrame
            summary_data = []
            for pii_type, count in pii_summary.items():
                avg_conf = sum(confidence_by_type[pii_type]) / len(confidence_by_type[pii_type])
                summary_data.append({
                    'PII Type': pii_type.replace('_', ' ').title(),
                    'Total Count': count,
                    'Avg Confidence': f"{avg_conf:.2f}",
                    'Documents': len(confidence_by_type[pii_type])
                })
            
            summary_df = pd.DataFrame(summary_data).sort_values('Total Count', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(summary_df, use_container_width=True)
            
            with col2:
                # Create chart data
                chart_data = summary_df[['PII Type', 'Total Count']].set_index('PII Type')
                st.bar_chart(chart_data)
        
        # Advanced filtering and display
        st.subheader("🔎 Detailed Analysis")
        
        # Enhanced filter options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            selected_pii_types = st.multiselect(
                "Filter by PII Type",
                options=list(pii_summary.keys()),
                default=list(pii_summary.keys())[:5] if len(pii_summary) > 5 else list(pii_summary.keys()),
                help="Select specific PII types to display"
            )
        
        with col2:
            min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.3, help="Filter results by confidence")
            
        with col3:
            sort_by = st.selectbox("Sort by", ["Document ID", "PII Count", "Confidence"], index=2)
        
        # Additional display options
        show_doc_preview = st.checkbox("Show Document Preview", value=True)
        show_confidence_score = st.checkbox("Show Confidence Scores", value=True)
        highlight_high_confidence = st.checkbox("Highlight High Confidence Items", value=True)
        
        # Sort results
        if sort_by == "PII Count":
            sorted_results = sorted(st.session_state.extracted_data, key=lambda x: x['pii_count'], reverse=True)
        elif sort_by == "Confidence":
            sorted_results = sorted(st.session_state.extracted_data, key=lambda x: x['confidence_score'], reverse=True)
        else:
            sorted_results = st.session_state.extracted_data
        
        # Display filtered and sorted results
        displayed_count = 0
        for i, result in enumerate(sorted_results):
            
            # Apply confidence filter
            if result['confidence_score'] < min_confidence:
                continue
            
            # Filter based on selected PII types
            filtered_pii = {k: v for k, v in result['pii_found'].items() if k in selected_pii_types}
            
            if not filtered_pii:
                continue
            
            displayed_count += 1
            
            # Enhanced expander title
            confidence_indicator = "🟢" if result['confidence_score'] > 0.8 else "🟡" if result['confidence_score'] > 0.5 else "🔴"
            title = f"{confidence_indicator} Document {result['document_id']} - {result['pii_count']} PII items"
            if show_confidence_score:
                title += f" (Confidence: {result['confidence_score']:.2f})"
            
            with st.expander(title):
                
                if show_doc_preview:
                    st.text_area("📄 Document Preview", result['document_preview'], height=120, disabled=True)
                
                # Display metadata
                if show_confidence_score:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence Score", f"{result['confidence_score']:.2f}")
                    col2.metric("PII Items Found", result['pii_count'])
                    col3.metric("Text Length", result['original_text_length'])
                
                # Display PII by category with enhanced formatting
                for pii_type, values in filtered_pii.items():
                    if values:
                        # Determine display style based on confidence and type
                        is_high_conf = result['confidence_score'] > 0.8
                        header_style = "**" if not highlight_high_confidence else "**🔥 " if is_high_conf else "**"
                        
                        st.write(f"{header_style}{pii_type.replace('_', ' ').title()}:**")
                        
                        # Special handling for different data types
                        if pii_type == 'validated_phones':
                            for phone_data in values:
                                if phone_data.get('valid', False):
                                    st.success(f"📱 {phone_data['formatted']} - {phone_data.get('country', 'Unknown')} ({phone_data.get('carrier', 'Unknown carrier')})")
                                else:
                                    st.warning(f"📱 {phone_data['original']} - Validation failed")
                                    
                        elif pii_type == 'validated_emails':
                            for email_data in values:
                                if email_data.get('valid', False):
                                    st.success(f"📧 {email_data['normalized']}")
                                else:
                                    st.warning(f"📧 {email_data['original']} - Validation failed")
                                    
                        elif pii_type in ['aadhaar', 'aadhaar_verbose']:
                            for value in values:
                                # Mask Aadhaar for security
                                masked = f"XXXX-XXXX-{str(value)[-4:]}" if len(str(value)) >= 4 else "XXXX-XXXX-XXXX"
                                st.info(f"🆔 {masked} (Aadhaar)")
                                
                        elif pii_type in ['pan', 'pan_verbose']:
                            for value in values:
                                # Mask PAN
                                masked = f"{str(value)[:3]}XX{str(value)[-4:]}" if len(str(value)) >= 7 else "XXXXX1234X"
                                st.info(f"🆔 {masked} (PAN)")
                                
                        elif pii_type in ['extracted_names', 'ner_persons']:
                            for name in values:
                                if is_high_conf and highlight_high_confidence:
                                    st.success(f"👤 {name}")
                                else:
                                    st.write(f"👤 {name}")
                                    
                        elif pii_type in ['extracted_addresses']:
                            for address in values:
                                # Truncate long addresses
                                display_addr = address[:100] + "..." if len(address) > 100 else address
                                st.write(f"🏠 {display_addr}")
                                
                        elif pii_type in ['fir_number', 'case_number']:
                            for value in values:
                                st.info(f"📋 {value}")
                                
                        else:
                            # Default display for other types
                            for value in values:
                                icon = "🔥" if is_high_conf and highlight_high_confidence else "•"
                                st.write(f"{icon} {value}")
                
                st.divider()
        
        st.info(f"Displayed {displayed_count} documents matching your filters.")
        
        # Enhanced export functionality
        st.subheader("📤 Export Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Export Detailed JSON"):
                json_data = json.dumps(st.session_state.extracted_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Detailed JSON",
                    data=json_data,
                    file_name=f"pii_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📋 Export Summary CSV"):
                # Create summary CSV
                csv_data = []
                for result in st.session_state.extracted_data:
                    for pii_type, values in result['pii_found'].items():
                        if isinstance(values, list):
                            for value in values:
                                csv_data.append({
                                    'Document_ID': result['document_id'],
                                    'PII_Type': pii_type,
                                    'PII_Value': str(value)[:100],  # Truncate for CSV
                                    'Confidence': result['confidence_score'],
                                    'Document_Preview': result['document_preview'][:150]
                                })
                        else:
                            csv_data.append({
                                'Document_ID': result['document_id'],
                                'PII_Type': pii_type,
                                'PII_Value': str(values)[:100],
                                'Confidence': result['confidence_score'],
                                'Document_Preview': result['document_preview'][:150]
                            })
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    csv_string = df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary CSV",
                        data=csv_string,
                        file_name=f"pii_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("🔒 Export Masked Data"):
                # Create masked version for sharing
                masked_data = []
                for result in st.session_state.extracted_data:
                    masked_pii = {}
                    for pii_type, values in result['pii_found'].items():
                        if pii_type in ['aadhaar', 'aadhaar_verbose']:
                            masked_pii[pii_type] = [f"XXXX-XXXX-{str(v)[-4:]}" for v in values]
                        elif pii_type in ['pan', 'pan_verbose']:
                            masked_pii[pii_type] = [f"{str(v)[:3]}XX{str(v)[-4:]}" for v in values]
                        elif pii_type == 'validated_phones':
                            masked_pii[pii_type] = [f"XXXXX{p['original'][-5:]}" if p.get('valid') else "XXXXXXXXXX" for p in values]
                        else:
                            masked_pii[pii_type] = values
                    
                    masked_data.append({
                        'document_id': result['document_id'],
                        'pii_found': masked_pii,
                        'confidence_score': result['confidence_score'],
                        'pii_count': result['pii_count']
                    })
                
                json_masked = json.dumps(masked_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Masked JSON",
                    data=json_masked,
                    file_name=f"pii_masked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col4:
            if st.button("🗑️ Clear Results"):
                st.session_state.extracted_data = None
                st.rerun()
    
    # Enhanced manual text input for testing OCR documents
    st.subheader("✏️ Manual OCR Text Analysis")
    st.markdown("**Test PII extraction on your OCR text samples**")
    
    # Provide sample OCR text
    if st.button("Load Sample OCR Text"):
        sample_text = """P.S. (Police Thane): Bhosari 
FIR No. ( ⁇  M Khab Đ.): 0523 
Date and Time of FIR ( ⁇ . True. Dated time): 
19/11/2017 at 21:33 
District (Jesus ã¡a): Pune City 
Year (Wash [): 2017
Name: राम कुमार शर्मा
Mobile: +91-9876543210
Address: शोभा हेट Íया Sad ⁇ ma, Moya M ⁇ at, Ashita, Asaravadi, Pune, 411001
Aadhaar: 1234-5678-9012
PAN: ABCDE1234F"""
        st.session_state['manual_text'] = sample_text
    
    manual_text = st.text_area(
        "Enter OCR text for PII extraction",
        value=st.session_state.get('manual_text', ''),
        placeholder="Paste your OCR-processed Hindi/English text here...",
        height=200,
        help="This tool is optimized for noisy OCR text with artifacts"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🧹 Clean & Analyze Text") and manual_text:
            with st.spinner("Processing OCR text..."):
                # Show cleaning process
                st.subheader("🔄 Text Cleaning Process")
                
                original_length = len(manual_text)
                cleaned_text = st.session_state.pii_extractor.text_cleaner.clean_text(manual_text)
                cleaned_length = len(cleaned_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Text:**")
                    st.text_area("", manual_text, height=100, disabled=True)
                    st.metric("Original Length", original_length)
                    
                with col2:
                    st.write("**Cleaned Text:**")
                    st.text_area("", cleaned_text, height=100, disabled=True)
                    st.metric("Cleaned Length", cleaned_length)
                
                # Extract PII
                pii_results = st.session_state.pii_extractor.extract_all_pii(manual_text)
                
                st.subheader("🎯 Extracted PII")
                if pii_results:
                    for pii_type, values in pii_results.items():
                        st.write(f"**{pii_type.replace('_', ' ').title()}:**")
                        
                        if pii_type == 'validated_phones':
                            for phone_data in values:
                                if phone_data.get('valid', False):
                                    st.success(f"📱 {phone_data['formatted']} ✅")
                                else:
                                    st.warning(f"📱 {phone_data['original']} ❌")
                        elif pii_type == 'validated_emails':
                            for email_data in values:
                                if email_data.get('valid', False):
                                    st.success(f"📧 {email_data['normalized']} ✅")
                                else:
                                    st.warning(f"📧 {email_data['original']} ❌")
                        else:
                            for value in values:
                                st.write(f"• {value}")
                        st.write("")
                else:
                    st.info("No PII found in the text.")
    
    with col2:
        if st.button("📊 Analyze Text Quality") and manual_text:
            # Text quality analysis
            st.subheader("📊 OCR Quality Analysis")
            
            # Calculate OCR error indicators
            total_chars = len(manual_text)
            special_chars = len(re.findall(r'[⁇ã¡aĐÍÖ¡⇢èğç\\]', manual_text))
            ocr_error_rate = (special_chars / total_chars * 100) if total_chars > 0 else 0
            
            # Language detection
            hindi_chars = len(re.findall(r'[\u0900-\u097F]', manual_text))
            english_chars = len(re.findall(r'[a-zA-Z]', manual_text))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("OCR Error Rate", f"{ocr_error_rate:.1f}%")
            col2.metric("Hindi Characters", hindi_chars)
            col3.metric("English Characters", english_chars)
            
            # Text composition pie chart data
            if hindi_chars + english_chars > 0:
                lang_data = pd.DataFrame({
                    'Language': ['Hindi', 'English'],
                    'Characters': [hindi_chars, english_chars]
                })
                st.bar_chart(lang_data.set_index('Language'))
            
            # Quality recommendations
            st.subheader("💡 Quality Recommendations")
            if ocr_error_rate > 20:
                st.error("High OCR error rate detected. Consider re-processing the source document.")
            elif ocr_error_rate > 10:
                st.warning("Moderate OCR errors detected. Results may need manual verification.")
            else:
                st.success("Good text quality detected.")

# Enhanced installation requirements
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Installation Requirements")

with st.sidebar.expander("Package Installation"):
    st.code("""
# Core packages
pip install streamlit pandas

# Elasticsearch
pip install elasticsearch

# NLP packages  
pip install spacy transformers torch
python -m spacy download en_core_web_sm

# Validation packages
pip install phonenumbers email-validator

# Optional for better performance
pip install accelerate optimum
""")

with st.sidebar.expander("Hardware Recommendations"):
    st.markdown("""
    **For large datasets:**
    - RAM: 8GB+ recommended
    - CPU: Multi-core for faster processing
    - GPU: Optional, helps with NER models
    
    **For basic usage:**
    - RAM: 4GB minimum
    - Any modern CPU works fine
    """)

if __name__ == "__main__":
    main()
