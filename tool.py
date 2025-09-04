import streamlit as st
import pandas as pd
import json
from datetime import datetime
import tempfile
import os
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Core libraries for PDF processing and NLP
try:
    import fitz  # PyMuPDF for PDF text extraction
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    
    # Advanced NLP libraries
    import spacy
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        AutoModelForSequenceClassification, pipeline,
        BertTokenizer, BertForTokenClassification
    )
    import torch
    from sentence_transformers import SentenceTransformer
    import google.generativeai as genai
    from langdetect import detect, LangDetectError
    
    # Additional utilities
    import re
    from collections import defaultdict
    import unicodedata
    
except ImportError as e:
    st.error(f"Missing required library: {e}. Please install all dependencies.")

class AdvancedFIRExtractor:
    def __init__(self):
        """Initialize the FIR extractor with advanced NLP models."""
        self.setup_models()
        self.setup_indian_patterns()
        
    def setup_models(self):
        """Initialize all NLP models and tools."""
        try:
            # Load multilingual NER model
            self.ner_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                "xlm-roberta-base", num_labels=9
            )
            
            # Initialize spaCy for English and Hindi
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("English spaCy model not found. Using basic processing.")
                self.nlp_en = None
                
            try:
                self.nlp_hi = spacy.load("hi_core_news_sm")
            except OSError:
                st.warning("Hindi spaCy model not found. Using multilingual approach.")
                self.nlp_hi = None
            
            # Sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Classification pipeline for legal text
            self.classifier = pipeline(
                "text-classification",
                model="nlpaueb/legal-bert-base-uncased",
                return_all_scores=True
            )
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            self.ner_tokenizer = self.ner_model = None
            self.nlp_en = self.nlp_hi = None
    
    def setup_indian_patterns(self):
        """Setup patterns specific to Indian legal documents."""
        self.legal_acts = {
            'ipc': ['IPC', 'Indian Penal Code', 'भारतीय दंड संहिता', '1860'],
            'crpc': ['CrPC', 'Code of Criminal Procedure', 'दंड प्रक्रिया संहिता', '1973'],
            'evidence_act': ['Evidence Act', 'साक्ष्य अधिनियम', '1872'],
            'domestic_violence': ['Domestic Violence Act', 'घरेलू हिंसा अधिनियम', '2005'],
            'sc_st': ['SC/ST Act', 'अनुसूचित जाति अधिनियम', '1989'],
            'pocso': ['POCSO', 'Protection of Children', '2012'],
        }
        
        self.indian_states = [
            'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
            'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
            'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
            'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
            'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
            'Delhi', 'Jammu and Kashmir', 'Ladakh'
        ]
        
        self.jurisdiction_types = [
            'PAN_INDIA', 'STATE_LEVEL', 'DISTRICT_LEVEL', 'LOCAL', 'SPECIAL'
        ]
        
        # Enhanced patterns for Indian names and addresses
        self.name_indicators = [
            'complainant', 'accused', 'victim', 'applicant', 'respondent',
            'शिकायतकर्ता', 'आरोपी', 'पीड़ित', 'आवेदक'
        ]
        
        self.address_indicators = [
            'address', 'residence', 'village', 'district', 'pin', 'pincode',
            'पता', 'निवास', 'गांव', 'जिला', 'पिन'
        ]

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Advanced PDF text extraction using multiple techniques.
        """
        text = ""
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Method 1: PyMuPDF for direct text extraction
            doc = fitz.open(tmp_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            
            # Method 2: OCR if direct extraction yields poor results
            if len(text.strip()) < 100 or self.is_text_corrupted(text):
                st.info("Direct extraction yielded poor results. Using OCR...")
                text = self.ocr_extract_text(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return ""
        
        return self.clean_extracted_text(text)
    
    def is_text_corrupted(self, text: str) -> bool:
        """Check if extracted text is corrupted or contains too many special characters."""
        if not text:
            return True
        
        # Calculate ratio of alphanumeric characters
        alphanumeric_count = sum(c.isalnum() for c in text)
        total_count = len(text)
        
        if total_count == 0:
            return True
        
        ratio = alphanumeric_count / total_count
        return ratio < 0.5  # If less than 50% are alphanumeric, consider corrupted
    
    def ocr_extract_text(self, pdf_path: str) -> str:
        """Extract text using OCR with support for multiple Indian languages."""
        text = ""
        
        try:
            # Convert PDF to images
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Preprocess image for better OCR
                image = self.preprocess_image_for_ocr(image)
                
                # OCR with multiple languages
                custom_config = r'--oem 3 --psm 6 -l eng+hin+ben+guj+pan+tel+tam+kan+mal+ori'
                page_text = pytesseract.image_to_string(image, config=custom_config)
                text += page_text + "\n"
            
            doc.close()
            
        except Exception as e:
            st.error(f"OCR extraction failed: {e}")
        
        return text
    
    def preprocess_image_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy."""
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to PIL
        return Image.fromarray(processed)
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\-.,;:()\[\]{}\/\\@#$%^&*+=|~`<>?]', ' ', text)
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of the text."""
        try:
            return detect(text)
        except LangDetectError:
            return 'unknown'
    
    def extract_pii_advanced(self, text: str) -> Dict[str, Any]:
        """
        Extract PII using advanced NLP techniques including:
        - Named Entity Recognition
        - Semantic similarity
        - Context-aware extraction
        - Legal document understanding
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
            'oparty': [],
            'names': [],
            'addresses': [],
            'jurisdiction': None,
            'jurisdiction_type': None,
            'confidence_scores': {}
        }
        
        # Detect language
        language = self.detect_language(text)
        
        # Choose appropriate NLP model
        nlp_model = self.nlp_hi if language == 'hi' else self.nlp_en
        if nlp_model is None:
            # Fallback to multilingual approach
            nlp_model = self.nlp_en if self.nlp_en else None
        
        # Extract using different techniques
        results.update(self.extract_using_ner(text, nlp_model))
        results.update(self.extract_using_patterns(text))
        results.update(self.extract_using_semantic_similarity(text))
        results.update(self.extract_legal_entities(text))
        
        # Post-process and validate results
        results = self.validate_and_clean_results(results)
        
        return results
    
    def extract_using_ner(self, text: str, nlp_model) -> Dict[str, Any]:
        """Extract entities using Named Entity Recognition."""
        results = defaultdict(list)
        
        if nlp_model:
            doc = nlp_model(text[:1000000])  # Limit text length for processing
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    results['names'].append(ent.text.strip())
                elif ent.label_ in ["GPE", "LOC"]:
                    # Could be state, district, or address
                    if ent.text in self.indian_states:
                        results['state_name'] = ent.text
                    else:
                        results['addresses'].append(ent.text.strip())
                elif ent.label_ == "ORG":
                    # Could be police station
                    if any(keyword in ent.text.lower() for keyword in ['police', 'station', 'thana']):
                        results['police_station'] = ent.text.strip()
        
        return dict(results)
    
    def extract_using_patterns(self, text: str) -> Dict[str, Any]:
        """Extract information using intelligent pattern matching."""
        results = {}
        
        # FIR Number extraction
        fir_patterns = [
            r'FIR\s*No\.?\s*:?\s*(\d+/\d+)',
            r'प्राथमिकी\s*संख्या\s*:?\s*(\d+/\d+)',
            r'Case\s*No\.?\s*:?\s*(\d+/\d+)',
            r'मामला\s*संख्या\s*:?\s*(\d+/\d+)'
        ]
        
        for pattern in fir_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results['fir_no'] = match.group(1)
                break
        
        # Year extraction
        year_patterns = [
            r'(\d{4})',  # Simple 4-digit year
            r'Year\s*:?\s*(\d{4})',
            r'वर्ष\s*:?\s*(\d{4})'
        ]
        
        current_year = datetime.now().year
        for pattern in year_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                year = int(match)
                if 1900 <= year <= current_year:  # Valid year range
                    results['year'] = year
                    break
            if 'year' in results:
                break
        
        # Police Station extraction
        ps_patterns = [
            r'Police\s+Station\s*:?\s*([A-Za-z\s]+)',
            r'थाना\s*:?\s*([A-Za-z\u0900-\u097F\s]+)',
            r'P\.?S\.?\s*:?\s*([A-Za-z\s]+)'
        ]
        
        for pattern in ps_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results['police_station'] = match.group(1).strip()
                break
        
        # Legal sections extraction
        section_patterns = [
            r'Section\s*(\d+(?:,\s*\d+)*)',
            r'धारा\s*(\d+(?:,\s*\d+)*)',
            r'u/s\s*(\d+(?:,\s*\d+)*)',
            r'IPC\s*(\d+(?:,\s*\d+)*)'
        ]
        
        sections = set()
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                sections.update([s.strip() for s in match.split(',')])
        
        if sections:
            results['under_sections'] = list(sections)
        
        return results
    
    def extract_using_semantic_similarity(self, text: str) -> Dict[str, Any]:
        """Use semantic similarity to extract relevant information."""
        results = {}
        
        # Split text into sentences
        sentences = re.split(r'[।.!?]+', text)
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        # Define target concepts and their embeddings
        target_concepts = {
            'complainant_info': "complainant name address details",
            'accused_info': "accused person name details",
            'police_station': "police station thana jurisdiction",
            'case_details': "case registered under sections acts"
        }
        
        concept_embeddings = self.sentence_model.encode(list(target_concepts.values()))
        
        # Find most similar sentences for each concept
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(sentence_embeddings, concept_embeddings)
        
        for i, (concept, _) in enumerate(target_concepts.items()):
            most_similar_idx = np.argmax(similarities[:, i])
            similarity_score = similarities[most_similar_idx, i]
            
            if similarity_score > 0.5:  # Threshold for relevance
                relevant_sentence = sentences[most_similar_idx]
                results[f'{concept}_context'] = relevant_sentence
                results[f'{concept}_confidence'] = float(similarity_score)
        
        return results
    
    def extract_legal_entities(self, text: str) -> Dict[str, Any]:
        """Extract legal-specific entities and classifications."""
        results = {'under_acts': []}
        
        # Identify legal acts
        for act_type, variations in self.legal_acts.items():
            for variation in variations:
                if variation.lower() in text.lower():
                    results['under_acts'].append(f"{act_type.upper()} - {variation}")
        
        # Case category classification using semantic analysis
        case_categories = [
            'theft', 'assault', 'domestic violence', 'fraud', 'murder',
            'rape', 'kidnapping', 'robbery', 'cybercrime', 'corruption'
        ]
        
        # Use the legal BERT model for classification
        try:
            if len(text) > 0:
                classification_results = self.classifier(text[:512])  # BERT token limit
                
                # Find the most likely category
                best_category = max(classification_results[0], key=lambda x: x['score'])
                if best_category['score'] > 0.5:
                    results['revised_case_category'] = best_category['label']
                    results['category_confidence'] = best_category['score']
        except Exception as e:
            st.warning(f"Classification failed: {e}")
        
        # Jurisdiction detection
        for jurisdiction in self.jurisdiction_types:
            if jurisdiction.lower().replace('_', ' ') in text.lower():
                results['jurisdiction_type'] = jurisdiction
                break
        else:
            # Default jurisdiction logic
            if any(state in text for state in self.indian_states):
                results['jurisdiction_type'] = 'STATE_LEVEL'
            else:
                results['jurisdiction_type'] = 'LOCAL'
        
        return results
    
    def validate_and_clean_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted results."""
        
        # Remove duplicates from lists
        for key in ['names', 'addresses', 'under_acts', 'under_sections', 'oparty']:
            if key in results and isinstance(results[key], list):
                results[key] = list(set(results[key]))
        
        # Validate FIR number format
        if results.get('fir_no'):
            if not re.match(r'\d+/\d+', results['fir_no']):
                results['fir_no'] = None
        
        # Validate year
        if results.get('year'):
            current_year = datetime.now().year
            if not (1900 <= results['year'] <= current_year):
                results['year'] = None
        
        # Clean names (remove common words that might be misidentified as names)
        if results.get('names'):
            stop_words = {'the', 'and', 'of', 'in', 'at', 'by', 'for', 'with'}
            results['names'] = [name for name in results['names'] 
                              if name.lower() not in stop_words and len(name) > 2]
        
        # Merge similar keys (consolidate oparty and names)
        all_names = set()
        if results.get('names'):
            all_names.update(results['names'])
        if results.get('oparty'):
            all_names.update(results['oparty'])
        
        results['names'] = list(all_names)
        results['oparty'] = list(all_names)  # Assuming opposite party is among the names
        
        return results

def main():
    st.set_page_config(
        page_title="Advanced FIR PII Extraction Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 Advanced FIR PII Extraction Tool")
    st.markdown("""
    This tool uses state-of-the-art NLP techniques to extract Personal Identifiable Information (PII) 
    from FIR documents in multiple Indian languages.
    
    **Features:**
    - Multi-language support (Hindi, English, and other Indian languages)
    - Advanced OCR for scanned documents
    - Named Entity Recognition (NER)
    - Semantic similarity analysis
    - Legal document understanding
    - Context-aware extraction
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence score for extracted information"
        )
        
        show_debug_info = st.checkbox(
            "Show Debug Information",
            help="Display additional processing details"
        )
        
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "Excel"],
            help="Choose export format for results"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload FIR Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file containing FIR document"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Initialize extractor
            with st.spinner("Initializing advanced NLP models..."):
                extractor = AdvancedFIRExtractor()
            
            if st.button("🚀 Extract PII Information", type="primary"):
                with st.spinner("Processing document... This may take a few minutes."):
                    
                    # Extract text
                    progress_bar = st.progress(0)
                    st.info("Step 1/3: Extracting text from PDF...")
                    progress_bar.progress(33)
                    
                    extracted_text = extractor.extract_text_from_pdf(uploaded_file)
                    
                    if not extracted_text:
                        st.error("Failed to extract text from the PDF. Please check if the file is valid.")
                        return
                    
                    # Show extracted text preview
                    if show_debug_info:
                        with st.expander("📋 Extracted Text Preview"):
                            st.text_area(
                                "First 1000 characters:",
                                value=extracted_text[:1000],
                                height=200
                            )
                    
                    # Extract PII
                    st.info("Step 2/3: Analyzing text and extracting PII...")
                    progress_bar.progress(66)
                    
                    pii_results = extractor.extract_pii_advanced(extracted_text)
                    
                    progress_bar.progress(100)
                    st.success("✅ Processing completed!")
                    
                    # Display results in the second column
                    with col2:
                        st.header("📊 Extraction Results")
                        
                        # Create results summary
                        st.subheader("🎯 Key Information")
                        
                        key_info = {
                            "FIR Number": pii_results.get('fir_no', 'Not found'),
                            "Year": pii_results.get('year', 'Not found'),
                            "State": pii_results.get('state_name', 'Not found'),
                            "District": pii_results.get('dist_name', 'Not found'),
                            "Police Station": pii_results.get('police_station', 'Not found'),
                            "Jurisdiction Type": pii_results.get('jurisdiction_type', 'Not found')
                        }
                        
                        for key, value in key_info.items():
                            st.write(f"**{key}:** {value}")
                        
                        # Legal Information
                        st.subheader("⚖️ Legal Information")
                        
                        if pii_results.get('under_acts'):
                            st.write("**Acts:** " + ", ".join(pii_results['under_acts']))
                        else:
                            st.write("**Acts:** Not found")
                            
                        if pii_results.get('under_sections'):
                            st.write("**Sections:** " + ", ".join(pii_results['under_sections']))
                        else:
                            st.write("**Sections:** Not found")
                            
                        if pii_results.get('revised_case_category'):
                            st.write(f"**Case Category:** {pii_results['revised_case_category']}")
                        else:
                            st.write("**Case Category:** Not determined")
                        
                        # Personal Information
                        st.subheader("👤 Personal Information")
                        
                        if pii_results.get('names'):
                            st.write("**Names Found:**")
                            for name in pii_results['names']:
                                st.write(f"- {name}")
                        else:
                            st.write("**Names:** None found")
                            
                        if pii_results.get('addresses'):
                            st.write("**Addresses Found:**")
                            for address in pii_results['addresses']:
                                st.write(f"- {address}")
                        else:
                            st.write("**Addresses:** None found")
                        
                        # Confidence Scores
                        if show_debug_info and pii_results.get('confidence_scores'):
                            st.subheader("📈 Confidence Scores")
                            for key, score in pii_results['confidence_scores'].items():
                                st.write(f"**{key}:** {score:.2f}")
                        
                        # Export functionality
                        st.subheader("💾 Export Results")
                        
                        if export_format == "JSON":
                            json_data = json.dumps(pii_results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name=f"fir_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        elif export_format == "CSV":
                            # Flatten the results for CSV
                            flat_results = {}
                            for key, value in pii_results.items():
                                if isinstance(value, list):
                                    flat_results[key] = "; ".join(map(str, value))
                                else:
                                    flat_results[key] = str(value) if value is not None else ""
                            
                            df = pd.DataFrame([flat_results])
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv_data,
                                file_name=f"fir_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        # Display full results in expandable section
                        with st.expander("🔍 View Full Results"):
                            st.json(pii_results)
    
    # Instructions and tips
    with st.expander("💡 Usage Tips and Requirements"):
        st.markdown("""
        ### System Requirements:
        - Install required packages: `pip install streamlit PyMuPDF pytesseract pillow opencv-python spacy transformers torch sentence-transformers google-generativeai langdetect scikit-learn`
        - Download spaCy models: `python -m spacy download en_core_web_sm hi_core_news_sm`
        - Install Tesseract OCR with language packs
        
        ### Tips for Better Results:
        - Upload clear, high-quality PDF files
        - Ensure the document contains text (not just images)
        - For scanned documents, higher resolution yields better OCR results
        - The tool works best with standard FIR formats
        
        ### Supported Languages:
        - English
        - Hindi (हिंदी)
        - Bengali (বাংলা)
        - Gujarati (ગુજરાતી)
        - Tamil (தமிழ்)
        - Telugu (తెలుగు)
        - Kannada (ಕನ್ನಡ)
        - Malayalam (മലയാളം)
        - Punjabi (ਪੰਜਾਬੀ)
        - Odia (ଓଡ଼ିଆ)
        """)

if __name__ == "__main__":
    main()
