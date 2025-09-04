import streamlit as st
import pandas as pd
import json
import io
import base64
from datetime import datetime
import tempfile
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core libraries for PDF processing and NLP
try:
    import fitz  # PyMuPDF for PDF text extraction
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from langdetect import detect, LangDetectError
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Standard library imports
import re
from collections import defaultdict, Counter
import unicodedata
import string

class FallbackPDFExtractor:
    """Fallback PDF extractor using built-in libraries when PyMuPDF is not available."""
    
    @staticmethod
    def extract_text_simple(pdf_bytes: bytes) -> str:
        """Simple text extraction fallback."""
        try:
            # Try to find readable text in the PDF bytes
            text_content = pdf_bytes.decode('latin-1', errors='ignore')
            
            # Look for text between stream objects
            text_matches = re.findall(r'stream\s*(.*?)\s*endstream', text_content, re.DOTALL)
            extracted_text = ""
            
            for match in text_matches:
                # Clean and extract readable text
                cleaned = re.sub(r'[^\x20-\x7E\u0900-\u097F]', ' ', match)
                extracted_text += cleaned + " "
            
            return extracted_text.strip()
        except Exception:
            return ""

class AdvancedFIRExtractor:
    def __init__(self):
        """Initialize the FIR extractor with available models."""
        self.models_loaded = False
        self.setup_patterns()
        self.setup_models()
        
    def setup_patterns(self):
        """Setup comprehensive patterns for Indian legal documents."""
        
        # Legal acts patterns
        self.legal_acts = {
            'ipc': [
                'IPC', 'Indian Penal Code', 'भारतीय दंड संहिता', '1860',
                'I.P.C', 'Penal Code', 'दंड संहिता'
            ],
            'crpc': [
                'CrPC', 'Cr.P.C', 'Code of Criminal Procedure', 
                'दंड प्रक्रिया संहिता', '1973', 'Criminal Procedure'
            ],
            'evidence_act': [
                'Evidence Act', 'साक्ष्य अधिनियम', '1872',
                'Indian Evidence Act', 'Evidence'
            ],
            'domestic_violence': [
                'Domestic Violence Act', 'घरेलू हिंसा अधिनियम', '2005',
                'Protection of Women', 'DV Act'
            ],
            'sc_st': [
                'SC/ST Act', 'अनुसूचित जाति अधिनियम', '1989',
                'Scheduled Castes', 'Scheduled Tribes', 'Atrocities Act'
            ],
            'pocso': [
                'POCSO', 'Protection of Children', '2012',
                'POCSO Act', 'Sexual Offences'
            ],
            'ndps': [
                'NDPS', 'Narcotic Drugs', 'Psychotropic Substances',
                'मादक पदार्थ', '1985'
            ]
        }
        
        # Indian states and UTs
        self.indian_states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
            'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
            'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
            'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
            'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Puducherry', 'Chandigarh',
            'Andaman and Nicobar Islands', 'Dadra and Nagar Haveli', 'Daman and Diu',
            'Lakshadweep'
        ]
        
        # Common Indian districts (sample)
        self.indian_districts = [
            'Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara',
            'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia', 'Hingoli',
            'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Mumbai City', 'Mumbai Suburban',
            'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad', 'Palghar',
            'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara',
            'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim', 'Yavatmal'
        ]
        
        # Case categories
        self.case_categories = {
            'theft': ['theft', 'चोरी', 'stealing', 'stolen', 'robbery'],
            'assault': ['assault', 'हमला', 'attack', 'beating', 'violence'],
            'fraud': ['fraud', 'धोखाधड़ी', 'cheating', 'forgery', 'scam'],
            'domestic_violence': ['domestic violence', 'घरेलू हिंसा', 'dowry', 'harassment'],
            'murder': ['murder', 'हत्या', 'killing', 'homicide', 'death'],
            'rape': ['rape', 'बलात्कार', 'sexual assault', 'molestation'],
            'kidnapping': ['kidnapping', 'अपहरण', 'abduction', 'missing person'],
            'cybercrime': ['cyber', 'online fraud', 'internet', 'digital', 'hacking'],
            'corruption': ['corruption', 'भ्रष्टाचार', 'bribery', 'illegal gratification'],
            'drugs': ['drugs', 'narcotics', 'substance', 'NDPS', 'illegal possession']
        }
        
        # Multi-language patterns
        self.patterns = {
            'fir_number': [
                r'(?i)fir\s*(?:no\.?|number)\s*:?\s*(\d+/\d+)',
                r'(?i)प्राथमिकी\s*संख्या\s*:?\s*(\d+/\d+)',
                r'(?i)case\s*(?:no\.?|number)\s*:?\s*(\d+/\d+)',
                r'(?i)मामला\s*संख्या\s*:?\s*(\d+/\d+)',
                r'(?i)f\.?i\.?r\.?\s*(?:no\.?)\s*:?\s*(\d+/\d+)',
                r'(\d+/\d{4})',  # Generic format
                r'(\d{1,4}/\d{4})'  # Year format
            ],
            'year': [
                r'(?i)year\s*:?\s*(\d{4})',
                r'(?i)वर्ष\s*:?\s*(\d{4})',
                r'dated?\s*:?\s*\d{1,2}[/-]\d{1,2}[/-](\d{4})',
                r'(\d{4})',  # Simple 4-digit year
            ],
            'police_station': [
                r'(?i)police\s*station\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)थाना\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)p\.?s\.?\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)station\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)thana\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)'
            ],
            'sections': [
                r'(?i)section\s*(\d+(?:,\s*\d+)*)',
                r'(?i)धारा\s*(\d+(?:,\s*\d+)*)',
                r'(?i)u/s\s*(\d+(?:,\s*\d+)*)',
                r'(?i)sec\.?\s*(\d+(?:,\s*\d+)*)',
                r'(?i)ipc\s*(\d+(?:,\s*\d+)*)',
                r'(?i)under\s*sections?\s*(\d+(?:,\s*\d+)*)'
            ],
            'complainant': [
                r'(?i)complainant\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)शिकायतकर्ता\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)applicant\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)informant\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)'
            ],
            'accused': [
                r'(?i)accused\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)आरोपी\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)respondent\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)',
                r'(?i)suspect\s*:?\s*([A-Za-z\s\u0900-\u097F]+?)(?:\n|$|,)'
            ],
            'address': [
                r'(?i)address\s*:?\s*([A-Za-z0-9\s\u0900-\u097F,.-]+?)(?:\n\n|police|station)',
                r'(?i)पता\s*:?\s*([A-Za-z0-9\s\u0900-\u097F,.-]+?)(?:\n\n|police|station)',
                r'(?i)residence\s*:?\s*([A-Za-z0-9\s\u0900-\u097F,.-]+?)(?:\n\n|police|station)',
                r'(?i)निवास\s*:?\s*([A-Za-z0-9\s\u0900-\u097F,.-]+?)(?:\n\n|police|station)'
            ],
            'pincode': [
                r'(?i)pin\s*(?:code)?\s*:?\s*(\d{6})',
                r'(?i)पिन\s*कोड\s*:?\s*(\d{6})',
                r'\b(\d{6})\b'  # 6-digit number
            ]
        }
        
    def setup_models(self):
        """Initialize available NLP models."""
        self.models = {}
        
        # Initialize spaCy models if available
        if SPACY_AVAILABLE:
            try:
                import spacy
                try:
                    self.models['nlp_en'] = spacy.load("en_core_web_sm")
                except OSError:
                    try:
                        self.models['nlp_en'] = spacy.load("en_core_web_md")
                    except OSError:
                        self.models['nlp_en'] = None
                
                try:
                    self.models['nlp_hi'] = spacy.load("hi_core_news_sm")
                except OSError:
                    self.models['nlp_hi'] = None
                    
            except Exception as e:
                logger.warning(f"SpaCy initialization failed: {e}")
        
        # Initialize sentence transformers if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Sentence transformer initialization failed: {e}")
                self.models['sentence_model'] = None
        
        # Initialize classification pipeline if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.models['classifier'] = pipeline(
                    "text-classification",
                    model="nlpaueb/legal-bert-base-uncased",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Legal BERT initialization failed: {e}")
                self.models['classifier'] = None
        
        self.models_loaded = True
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using multiple fallback methods."""
        text = ""
        
        try:
            pdf_bytes = pdf_file.getvalue()
            
            # Method 1: PyMuPDF (preferred)
            if PYMUPDF_AVAILABLE:
                text = self._extract_with_pymupdf(pdf_bytes)
                if len(text.strip()) > 50 and not self._is_text_corrupted(text):
                    return self._clean_extracted_text(text)
            
            # Method 2: OCR if available and needed
            if OCR_AVAILABLE and (not text or len(text.strip()) < 50 or self._is_text_corrupted(text)):
                st.info("Direct extraction yielded poor results. Using OCR...")
                ocr_text = self._extract_with_ocr(pdf_bytes)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            
            # Method 3: Fallback extraction
            if not text or len(text.strip()) < 20:
                st.info("Using fallback text extraction...")
                text = FallbackPDFExtractor.extract_text_simple(pdf_bytes)
            
        except Exception as e:
            st.error(f"Error in PDF processing: {e}")
            logger.error(f"PDF processing error: {e}")
        
        return self._clean_extracted_text(text)
    
    def _extract_with_pymupdf(self, pdf_bytes: bytes) -> str:
        """Extract text using PyMuPDF."""
        text = ""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text + "\n"
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
        return text
    
    def _extract_with_ocr(self, pdf_bytes: bytes) -> str:
        """Extract text using OCR."""
        text = ""
        try:
            # Convert PDF to images using PyMuPDF
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                for page_num in range(min(5, len(doc))):  # Limit to first 5 pages
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Preprocess for OCR
                    image = self._preprocess_image_for_ocr(image)
                    
                    # OCR with multiple language support
                    custom_config = r'--oem 3 --psm 6 -l eng+hin'
                    page_text = pytesseract.image_to_string(image, config=custom_config)
                    text += page_text + "\n"
                
                doc.close()
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
        
        return text
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            return Image.fromarray(thresh)
        
        except Exception:
            return image  # Return original if preprocessing fails
    
    def _is_text_corrupted(self, text: str) -> bool:
        """Check if extracted text appears to be corrupted."""
        if not text or len(text.strip()) < 10:
            return True
        
        # Calculate ratio of readable characters
        readable_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in string.punctuation)
        total_chars = len(text)
        
        if total_chars == 0:
            return True
        
        readable_ratio = readable_chars / total_chars
        return readable_ratio < 0.7
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except common punctuation
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of the text."""
        if not LANGDETECT_AVAILABLE:
            return 'unknown'
        
        try:
            # Use first 1000 characters for detection
            sample_text = text[:1000] if len(text) > 1000 else text
            return detect(sample_text)
        except (LangDetectError, Exception):
            return 'unknown'
    
    def extract_pii_comprehensive(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive PII extraction using multiple techniques.
        """
        results = {
            'fir_no': None,
            'year': None,
            'state_name': None,
            'dist_name': None,
            'police_station': None,
            'under_acts': [],
            'under_sections': [],
            'revised_case_category': None,
            'complainant': None,
            'accused': [],
            'names': [],
            'addresses': [],
            'pincodes': [],
            'jurisdiction_type': None,
            'confidence_scores': {},
            'language_detected': self.detect_language(text)
        }
        
        # Pattern-based extraction (most reliable)
        pattern_results = self._extract_using_patterns(text)
        results.update(pattern_results)
        
        # Named Entity Recognition if available
        if self.models.get('nlp_en') or self.models.get('nlp_hi'):
            ner_results = self._extract_using_ner(text)
            results = self._merge_results(results, ner_results)
        
        # Semantic analysis if available
        if self.models.get('sentence_model') and SKLEARN_AVAILABLE:
            semantic_results = self._extract_using_semantic_analysis(text)
            results = self._merge_results(results, semantic_results)
        
        # Legal classification if available
        if self.models.get('classifier'):
            legal_results = self._extract_legal_information(text)
            results = self._merge_results(results, legal_results)
        
        # Context-based extraction
        context_results = self._extract_using_context(text)
        results = self._merge_results(results, context_results)
        
        # Post-process and validate
        results = self._validate_and_clean_results(results, text)
        
        return results
    
    def _extract_using_patterns(self, text: str) -> Dict[str, Any]:
        """Extract information using regex patterns."""
        results = {}
        
        # Extract FIR number
        for pattern in self.patterns['fir_number']:
            matches = re.findall(pattern, text)
            if matches:
                results['fir_no'] = matches[0]
                break
        
        # Extract year
        current_year = datetime.now().year
        for pattern in self.patterns['year']:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    year = int(match)
                    if 1950 <= year <= current_year:
                        results['year'] = year
                        break
                except ValueError:
                    continue
            if results.get('year'):
                break
        
        # Extract police station
        for pattern in self.patterns['police_station']:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                ps_name = matches[0].strip()
                if len(ps_name) > 2 and len(ps_name) < 100:
                    results['police_station'] = ps_name
                    break
        
        # Extract sections
        sections = set()
        for pattern in self.patterns['sections']:
            matches = re.findall(pattern, text)
            for match in matches:
                section_nums = [s.strip() for s in match.split(',')]
                sections.update(section_nums)
        
        if sections:
            results['under_sections'] = sorted(list(sections), key=lambda x: int(x) if x.isdigit() else 999)
        
        # Extract complainant
        for pattern in self.patterns['complainant']:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                complainant = matches[0].strip()
                if len(complainant) > 2 and len(complainant) < 100:
                    results['complainant'] = complainant
                    break
        
        # Extract accused persons
        accused_list = []
        for pattern in self.patterns['accused']:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                accused = match.strip()
                if len(accused) > 2 and len(accused) < 100:
                    accused_list.append(accused)
        
        if accused_list:
            results['accused'] = list(set(accused_list))
        
        # Extract addresses
        addresses = []
        for pattern in self.patterns['address']:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                address = match.strip()
                if len(address) > 10 and len(address) < 500:
                    addresses.append(address)
        
        if addresses:
            results['addresses'] = addresses[:5]  # Limit to top 5
        
        # Extract pincodes
        pincodes = set()
        for pattern in self.patterns['pincode']:
            matches = re.findall(pattern, text)
            for match in matches:
                if match.isdigit() and len(match) == 6:
                    pincodes.add(match)
        
        if pincodes:
            results['pincodes'] = sorted(list(pincodes))
        
        return results
    
    def _extract_using_ner(self, text: str) -> Dict[str, Any]:
        """Extract entities using Named Entity Recognition."""
        results = {'names': [], 'addresses': []}
        
        # Choose appropriate model based on detected language
        nlp_model = None
        if self.models.get('nlp_hi') and 'hi' in text:
            nlp_model = self.models['nlp_hi']
        elif self.models.get('nlp_en'):
            nlp_model = self.models['nlp_en']
        
        if not nlp_model:
            return results
        
        try:
            # Process text in chunks to avoid memory issues
            chunk_size = 100000  # 100k characters per chunk
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            names = set()
            locations = set()
            
            for chunk in text_chunks[:3]:  # Process max 3 chunks
                doc = nlp_model(chunk)
                
                for ent in doc.ents:
                    if ent.label_ == "PERSON" and len(ent.text.strip()) > 2:
                        names.add(ent.text.strip())
                    elif ent.label_ in ["GPE", "LOC"] and len(ent.text.strip()) > 2:
                        locations.add(ent.text.strip())
            
            # Filter and validate names
            valid_names = []
            for name in names:
                if (len(name) > 2 and len(name) < 50 and 
                    not any(digit in name for digit in '0123456789') and
                    name.lower() not in ['police', 'station', 'court', 'case', 'fir']):
                    valid_names.append(name)
            
            results['names'] = valid_names[:10]  # Limit to top 10
            
            # Process locations
            addresses = []
            for location in locations:
                if location in self.indian_states:
                    results['state_name'] = location
                elif location in self.indian_districts:
                    results['dist_name'] = location
                else:
                    addresses.append(location)
            
            results['addresses'] = addresses[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
        
        return results
    
    def _extract_using_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Extract information using semantic similarity."""
        results = {}
        
        try:
            sentences = re.split(r'[.!?।]+', text)[:50]  # Limit sentences
            if not sentences:
                return results
            
            sentence_embeddings = self.models['sentence_model'].encode(sentences)
            
            # Define semantic targets
            targets = {
                'police_station': "police station thana law enforcement office",
                'case_details': "case registered under sections criminal law",
                'complainant': "complainant informant person filing complaint",
                'accused': "accused person suspect defendant criminal"
            }
            
            target_embeddings = self.models['sentence_model'].encode(list(targets.values()))
            
            # Calculate similarities
            similarities = cosine_similarity(sentence_embeddings, target_embeddings)
            
            # Extract relevant information
            for i, (target_key, _) in enumerate(targets.items()):
                max_sim_idx = np.argmax(similarities[:, i])
                max_similarity = similarities[max_sim_idx, i]
                
                if max_similarity > 0.3:  # Similarity threshold
                    relevant_sentence = sentences[max_sim_idx]
                    results[f'{target_key}_context'] = relevant_sentence
                    results[f'{target_key}_confidence'] = float(max_similarity)
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
        
        return results
    
    def _extract_legal_information(self, text: str) -> Dict[str, Any]:
        """Extract legal acts and case classification."""
        results = {'under_acts': []}
        
        # Extract legal acts
        text_lower = text.lower()
        for act_type, variations in self.legal_acts.items():
            for variation in variations:
                if variation.lower() in text_lower:
                    act_info = f"{act_type.upper().replace('_', ' ')} - {variation}"
                    results['under_acts'].append(act_info)
        
        # Remove duplicates
        results['under_acts'] = list(set(results['under_acts']))
        
        # Case category classification
        try:
            if self.models.get('classifier') and len(text.strip()) > 50:
                # Use first 512 characters for classification (BERT limit)
                classification_input = text[:512]
                classification_results = self.models['classifier'](classification_input)
                
                if classification_results and len(classification_results[0]) > 0:
                    best_category = max(classification_results[0], key=lambda x: x['score'])
                    if best_category['score'] > 0.4:
                        results['revised_case_category'] = best_category['label']
                        results['category_confidence'] = float(best_category['score'])
        
        except Exception as e:
            logger.error(f"Legal classification failed: {e}")
        
        # Fallback case category detection using keywords
        if not results.get('revised_case_category'):
            for category, keywords in self.case_categories.items():
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        results['revised_case_category'] = category.replace('_', ' ').title()
                        break
                if results.get('revised_case_category'):
                    break
        
        return results
    
    def _extract_using_context(self, text: str) -> Dict[str, Any]:
        """Extract information using contextual analysis."""
        results = {}
        
        # State detection using comprehensive search
        for state in self.indian_states:
            if state.lower() in text.lower():
                results['state_name'] = state
                break
        
        # District detection
        for district in self.indian_districts:
            if district.lower() in text.lower():
                results['dist_name'] = district
                break
        
        # Jurisdiction type determination
        if results.get('state_name'):
            results['jurisdiction_type'] = 'STATE_LEVEL'
        elif any(keyword in text.lower() for keyword in ['district', 'जिला']):
            results['jurisdiction_type'] = 'DISTRICT_LEVEL'
        elif any(keyword in text.lower() for keyword in ['local', 'municipal', 'panchayat']):
            results['jurisdiction_type'] = 'LOCAL'
        else:
            results['jurisdiction_type'] = 'LOCAL'  # Default
        
        return results
    
    def _merge_results(self, base_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently merge extraction results."""
        for key, value in new_results.items():
            if value is None or (isinstance(value, (list, str)) and not value):
                continue
            
            if key not in base_results or base_results[key] is None:
                base_results[key] = value
            elif isinstance(value, list) and isinstance(base_results[key], list):
                # Merge lists and remove duplicates
                combined = list(set(base_results[key] + value))
                base_results[key] = combined
            elif isinstance(value, str) and not base_results[key]:
                base_results[key] = value
        
        return base_results
    
    def _validate_and_clean_results(self, results: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Validate and clean extracted results."""
        
        # Clean and validate FIR number
        if results.get('fir_no'):
            fir_no = str(results['fir_no']).strip()
            if not re.match(r'\d+/\d{4}', fir_no):
                # Try to find a better match in original text
                better_match = re.search(r'(\d{1,4}/\d{4})', original_text)
                if better_match:
                    results['fir_no'] = better_match.group(1)
                else:
                    results['fir_no'] = None
        
        # Validate year
        if results.get('year'):
            current_year = datetime.now().year
            if not (1950 <= results['year'] <= current_year):
                results['year'] = None
        
        # Clean names
        if results.get('names'):
            cleaned_names = []
            stop_words = {'the', 'and', 'of', 'in', 'at', 'by', 'for', 'with', 'police', 'station', 'case', 'fir'}
            
            for name in results['names']:
                name = name.strip()
                if (len(name) > 2 and len(name) < 50 and 
                    name.lower() not in stop_words and
                    not re.match(r'^\d+$', name):  # Not just numbers
                    cleaned_names.append(name)
            
            results['names'] = list(set(cleaned_names))[:10]  # Limit and deduplicate
        
        # Clean police station name
        if results.get('police_station'):
            ps_name = results['police_station'].strip()
            # Remove common suffixes that might be included
            ps_name = re.sub(r'\s+(police\s+station|p\.?s\.?|thana), '', ps_name, flags=re.IGNORECASE)
            if len(ps_name) > 1:
                results['police_station'] = ps_name
            else:
                results['police_station'] = None
        
        # Clean sections
        if results.get('under_sections'):
            valid_sections = []
            for section in results['under_sections']:
                if re.match(r'^\d+[a-zA-Z]?, str(section).strip()):
                    valid_sections.append(str(section).strip())
            results['under_sections'] = sorted(valid_sections, key=lambda x: int(re.match(r'\d+', x).group()))
        
        # Clean addresses
        if results.get('addresses'):
            cleaned_addresses = []
            for address in results['addresses']:
                address = address.strip()
                if len(address) > 10 and len(address) < 300:
                    cleaned_addresses.append(address)
            results['addresses'] = cleaned_addresses[:5]
        
        # Consolidate people information
        all_people = set()
        if results.get('complainant'):
            all_people.add(results['complainant'])
        if results.get('accused'):
            all_people.update(results['accused'])
        if results.get('names'):
            all_people.update(results['names'])
        
        results['all_persons'] = list(all_people)
        
        # Calculate confidence scores
        confidence_factors = {
            'fir_no': 1.0 if results.get('fir_no') else 0.0,
            'year': 1.0 if results.get('year') else 0.0,
            'police_station': 0.8 if results.get('police_station') else 0.0,
            'sections': 0.7 if results.get('under_sections') else 0.0,
            'acts': 0.6 if results.get('under_acts') else 0.0,
            'people': 0.5 if results.get('names') else 0.0,
        }
        
        overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
        results['overall_confidence'] = round(overall_confidence, 2)
        results['confidence_scores'] = confidence_factors
        
        return results

def create_downloadable_file(data: Dict[str, Any], format_type: str) -> str:
    """Create downloadable file content."""
    
    if format_type == "JSON":
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    elif format_type == "CSV":
        # Flatten the data for CSV
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                flat_data[key] = "; ".join(map(str, value)) if value else ""
            elif isinstance(value, dict):
                flat_data[key] = str(value) if value else ""
            else:
                flat_data[key] = str(value) if value is not None else ""
        
        df = pd.DataFrame([flat_data])
        return df.to_csv(index=False)
    
    elif format_type == "Excel":
        # Create Excel-compatible data
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                flat_data[key] = "; ".join(map(str, value)) if value else ""
            elif isinstance(value, dict):
                flat_data[key] = str(value) if value else ""
            else:
                flat_data[key] = str(value) if value is not None else ""
        
        df = pd.DataFrame([flat_data])
        return df.to_excel(index=False, engine='openpyxl')
    
    return ""

def display_extraction_results(results: Dict[str, Any]):
    """Display extraction results in a formatted way."""
    
    # Key Information Section
    st.subheader("🎯 Key Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("FIR Number", results.get('fir_no', 'Not found'))
        st.metric("Year", results.get('year', 'Not found'))
        st.metric("Police Station", results.get('police_station', 'Not found'))
    
    with col2:
        st.metric("State", results.get('state_name', 'Not found'))
        st.metric("District", results.get('dist_name', 'Not found'))
        st.metric("Language", results.get('language_detected', 'Unknown'))
    
    # Legal Information Section
    st.subheader("⚖️ Legal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        if results.get('under_sections'):
            st.write("**Sections:**")
            for section in results['under_sections']:
                st.write(f"• Section {section}")
        else:
            st.write("**Sections:** Not found")
    
    with col2:
        if results.get('under_acts'):
            st.write("**Legal Acts:**")
            for act in results['under_acts']:
                st.write(f"• {act}")
        else:
            st.write("**Legal Acts:** Not found")
    
    if results.get('revised_case_category'):
        confidence = results.get('category_confidence', 0)
        st.write(f"**Case Category:** {results['revised_case_category']} (Confidence: {confidence:.2f})")
    
    # Personal Information Section
    st.subheader("👤 Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results.get('complainant'):
            st.write(f"**Complainant:** {results['complainant']}")
        
        if results.get('accused'):
            st.write("**Accused Persons:**")
            for person in results['accused']:
                st.write(f"• {person}")
    
    with col2:
        if results.get('all_persons'):
            st.write("**All Persons Mentioned:**")
            for person in results['all_persons'][:10]:  # Limit display
                st.write(f"• {person}")
    
    # Address Information
    if results.get('addresses'):
        st.subheader("📍 Address Information")
        for i, address in enumerate(results['addresses'][:3], 1):  # Show max 3 addresses
            st.write(f"**Address {i}:** {address}")
    
    if results.get('pincodes'):
        st.write(f"**PIN Codes:** {', '.join(results['pincodes'])}")
    
    # Confidence Metrics
    if results.get('overall_confidence'):
        st.subheader("📊 Extraction Confidence")
        st.progress(results['overall_confidence'])
        st.write(f"Overall Confidence: {results['overall_confidence']:.0%}")

def main():
    st.set_page_config(
        page_title="Advanced FIR PII Extraction Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Advanced FIR PII Extraction Tool</h1>
        <p>Powered by State-of-the-Art Natural Language Processing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview
    st.markdown("""
    ### 🚀 Features
    - **Multi-language Support**: Hindi, English, and other Indian languages
    - **Advanced OCR**: Handles scanned documents with poor text quality
    - **Intelligent Extraction**: Uses NER, semantic analysis, and pattern matching
    - **Legal Document Understanding**: Specialized for Indian legal documents
    - **Robust Fallbacks**: Multiple extraction methods ensure reliable results
    - **Comprehensive Validation**: Smart validation and confidence scoring
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        st.subheader("📊 Model Status")
        st.write(f"**PyMuPDF (PDF):** {'✅' if PYMUPDF_AVAILABLE else '❌'}")
        st.write(f"**OCR Support:** {'✅' if OCR_AVAILABLE else '❌'}")
        st.write(f"**SpaCy NLP:** {'✅' if SPACY_AVAILABLE else '❌'}")
        st.write(f"**Transformers:** {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
        st.write(f"**Sentence Trans:** {'✅' if SENTENCE_TRANSFORMERS_AVAILABLE else '❌'}")
        st.write(f"**Language Det:** {'✅' if LANGDETECT_AVAILABLE else '❌'}")
        
        st.divider()
        
        # Settings
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum confidence score for extracted information"
        )
        
        show_debug_info = st.checkbox(
            "Show Debug Information",
            help="Display additional processing details"
        )
        
        show_confidence_scores = st.checkbox(
            "Show Confidence Scores",
            value=True,
            help="Display confidence metrics for each extraction"
        )
        
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "Excel"],
            help="Choose export format for results"
        )
        
        st.divider()
        
        # Instructions
        with st.expander("📖 Quick Instructions"):
            st.markdown("""
            1. Upload a PDF file containing FIR document
            2. Click 'Extract PII Information' to process
            3. Review extracted information
            4. Download results in preferred format
            
            **Supported file types:** PDF
            **Max file size:** 200MB
            **Languages:** Hindi, English, and mixed documents
            """)
    
    # Main content
    st.header("📄 Upload FIR Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file containing FIR document",
        type=['pdf'],
        help="Upload a clear, readable PDF file for best results",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # File info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.markdown(f"""
        <div class="status-success">
            <strong>✅ File uploaded successfully!</strong><br>
            📄 Name: {uploaded_file.name}<br>
            📊 Size: {file_size:.2f} MB<br>
            🕒 Ready for processing
        </div>
        """, unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Extract PII Information", type="primary", use_container_width=True):
            # Initialize extractor
            with st.spinner("Initializing advanced NLP models..."):
                try:
                    extractor = AdvancedFIRExtractor()
                    st.success("✅ Models initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Model initialization failed: {e}")
                    return
            
            # Create progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract text
                    status_text.text("🔍 Step 1/4: Extracting text from PDF...")
                    progress_bar.progress(25)
                    
                    extracted_text = extractor.extract_text_from_pdf(uploaded_file)
                    
                    if not extracted_text or len(extracted_text.strip()) < 50:
                        st.error("❌ Failed to extract sufficient text from the PDF. Please ensure the file contains readable text.")
                        return
                    
                    st.success(f"✅ Extracted {len(extracted_text)} characters of text")
                    
                    # Step 2: Language detection
                    status_text.text("🔍 Step 2/4: Detecting language and preprocessing...")
                    progress_bar.progress(50)
                    
                    detected_language = extractor.detect_language(extracted_text)
                    st.info(f"🌐 Detected language: {detected_language}")
                    
                    # Step 3: Extract PII
                    status_text.text("🔍 Step 3/4: Analyzing document and extracting PII...")
                    progress_bar.progress(75)
                    
                    pii_results = extractor.extract_pii_comprehensive(extracted_text)
                    
                    # Step 4: Post-processing
                    status_text.text("🔍 Step 4/4: Validating and organizing results...")
                    progress_bar.progress(100)
                    
                    st.success("✅ Processing completed successfully!")
                    
                    # Clear progress indicators
                    progress_container.empty()
                    
                    # Display results
                    st.header("📊 Extraction Results")
                    
                    # Show extracted text preview if debug is enabled
                    if show_debug_info:
                        with st.expander("📋 Extracted Text Preview"):
                            st.text_area(
                                "First 2000 characters of extracted text:",
                                value=extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text,
                                height=200,
                                disabled=True
                            )
                    
                    # Display formatted results
                    display_extraction_results(pii_results)
                    
                    # Export section
                    st.header("💾 Export Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if export_format == "JSON":
                            json_data = json.dumps(pii_results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name=f"fir_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    with col2:
                        if export_format == "CSV":
                            csv_data = create_downloadable_file(pii_results, "CSV")
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=f"fir_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with col3:
                        # Raw results view
                        with st.expander("🔍 View Raw Results"):
                            st.json(pii_results)
                    
                    # Summary statistics
                    if show_confidence_scores:
                        st.header("📈 Extraction Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            found_count = sum(1 for key in ['fir_no', 'year', 'police_station', 'complainant'] 
                                            if pii_results.get(key))
                            st.metric("Key Fields Found", f"{found_count}/4")
                        
                        with col2:
                            sections_count = len(pii_results.get('under_sections', []))
                            st.metric("Legal Sections", sections_count)
                        
                        with col3:
                            acts_count = len(pii_results.get('under_acts', []))
                            st.metric("Legal Acts", acts_count)
                        
                        with col4:
                            people_count = len(pii_results.get('all_persons', []))
                            st.metric("Persons Found", people_count)
                        
                        # Confidence breakdown
                        if pii_results.get('confidence_scores'):
                            st.subheader("🎯 Confidence Breakdown")
                            confidence_df = pd.DataFrame(
                                list(pii_results['confidence_scores'].items()),
                                columns=['Field', 'Confidence']
                            )
                            confidence_df['Confidence'] = confidence_df['Confidence'].apply(lambda x: f"{x:.0%}")
                            st.dataframe(confidence_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {str(e)}")
                    logger.error(f"Processing error: {e}")
                    
                    if show_debug_info:
                        st.exception(e)
    
    # Additional information
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("💡 Tips for Better Results"):
            st.markdown("""
            ### Document Quality
            - Upload clear, high-resolution PDF files
            - Ensure text is readable and not corrupted
            - Avoid heavily compressed or low-quality scans
            
            ### Supported Content
            - Standard FIR formats from Indian police stations
            - Multi-language documents (Hindi + English)
            - Both typed and handwritten documents (via OCR)
            
            ### Best Practices
            - Files under 50MB process faster
            - Portrait orientation works better than landscape
            - Documents with clear section headers give better results
            """)
    
    with col2:
        with st.expander("🔧 Technical Requirements"):
            st.markdown(f"""
            ### Current System Status
            - **PDF Processing:** {'Available' if PYMUPDF_AVAILABLE else 'Limited (Fallback mode)'}
            - **OCR Support:** {'Available' if OCR_AVAILABLE else 'Not available'}
            - **NLP Models:** {'Available' if SPACY_AVAILABLE else 'Pattern-based fallback'}
            - **AI Classification:** {'Available' if TRANSFORMERS_AVAILABLE else 'Keyword-based fallback'}
            
            ### For Full Features Install
            ```bash
            pip install PyMuPDF pytesseract pillow opencv-python
            pip install spacy transformers torch sentence-transformers
            pip install langdetect scikit-learn
            python -m spacy download en_core_web_sm
            ```
            
            ### Performance Notes
            - Processing time: 30 seconds - 5 minutes
            - Memory usage: ~500MB - 2GB depending on models
            - Works offline once models are downloaded
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🔍 Advanced FIR PII Extraction Tool | Built with Streamlit & Advanced NLP</p>
        <p><em>Designed for Indian Legal Document Processing</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
