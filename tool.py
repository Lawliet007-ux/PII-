import streamlit as st
import pdfplumber
import pandas as pd
import re
import spacy
from spacy import displacy
from collections import defaultdict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
from langdetect import detect, DetectorFactory
import easyocr
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import io

# Set seed for language detection
DetectorFactory.seed = 0

# Set page configuration
st.set_page_config(
    page_title="ULTIMATE FIR PII Extraction Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'pii_data' not in st.session_state:
    st.session_state.pii_data = {}
if 'ocr_used' not in st.session_state:
    st.session_state.ocr_used = False

# Load models with caching - Using the best available models
@st.cache_resource
def load_ner_model():
    """Load the best multilingual NER model"""
    try:
        return pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="average")
    except:
        try:
            return pipeline("ner", model="Babelscape/wikineural-multilingual-ner", aggregation_strategy="average")
        except:
            return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource
def load_qa_model():
    """Load the best QA model"""
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2")
    except:
        return None

@st.cache_resource
def load_spacy_model():
    """Load the best spaCy model"""
    try:
        return spacy.load("en_core_web_lg")
    except:
        try:
            spacy.cli.download("en_core_web_lg")
            return spacy.load("en_core_web_lg")
        except:
            return None

@st.cache_resource
def load_easyocr_reader():
    """Load EasyOCR reader for multilingual text extraction"""
    return easyocr.Reader(['en', 'hi', 'mr', 'te', 'ta', 'bn', 'gu', 'kn', 'ml', 'pa', 'ur'])

@st.cache_resource
def load_legal_phrases():
    """Load Indian legal phrases and patterns"""
    return {
        'hindi_phrases': {
            'fir': ['प्रथम सूचना रिपोर्ट', 'एफआईआर', 'रिपोर्ट संख्या', 'थाना'],
            'sections': ['धारा', 'उपधारा', 'कलम'],
            'acts': ['आईपीसी', 'दंड प्रक्रिया संहिता', 'भारतीय दंड संहिता'],
            'police': ['पुलिस स्टेशन', 'थाना', 'कोतवाली']
        },
        'english_phrases': {
            'fir': ['FIR', 'First Information Report', 'Report No', 'Station'],
            'sections': ['Section', 'Sec', 'U/S'],
            'acts': ['IPC', 'Indian Penal Code', 'CrPC', 'Criminal Procedure Code'],
            'police': ['Police Station', 'PS', 'Station House']
        }
    }

def extract_text_advanced(pdf_file):
    """Advanced text extraction with multiple fallbacks"""
    text = ""
    st.session_state.ocr_used = False
    
    # Method 1: Try PyMuPDF for better text extraction
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        pdf_file.seek(0)
    except:
        text = ""
    
    # Method 2: Try pdfplumber if PyMuPDF failed or extracted little text
    if len(text.strip()) < 100:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            pdf_file.seek(0)
        except:
            text = ""
    
    # Method 3: Use OCR if both previous methods failed
    if len(text.strip()) < 100:
        try:
            reader = load_easyocr_reader()
            images = convert_from_bytes(pdf_file.read())
            for image in images:
                img_array = np.array(image)
                results = reader.readtext(img_array, detail=0)
                text += " ".join(results) + "\n"
            st.session_state.ocr_used = True
            pdf_file.seek(0)
        except Exception as e:
            st.error(f"OCR failed: {str(e)}")
    
    return text if text.strip() else None

def detect_language_advanced(text):
    """Advanced language detection with fallbacks"""
    try:
        return detect(text)
    except:
        # Check for Hindi characters
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi'
        return 'en'

def extract_fir_number_advanced(text, language):
    """Advanced FIR number extraction with multiple patterns"""
    patterns = {
        'en': [
            r'FIR\s*(?:No\.?|Number|NO\.?|N\.?|num\.?|Num\.?)?\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'F\.I\.R\.\s*(?:No\.?|Number|NO\.?|N\.?|num\.?|Num\.?)?\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'First Information Report\s*(?:No\.?|Number|NO\.?|N\.?|num\.?|Num\.?)?\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'Registration\s*(?:No\.?|Number|NO\.?|N\.?|num\.?|Num\.?)?\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'Report\s*(?:No\.?|Number|NO\.?|N\.?|num\.?|Num\.?)?\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
        ],
        'hi': [
            r'एफआईआर\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'प्रथम सूचना रिपोर्ट\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'रिपोर्ट संख्या\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
            r'एफ\.आई\.आर\.\s*[:\-\s]*\s*([A-Za-z0-9\/\-\.]+)',
        ]
    }
    
    for pattern in patterns.get(language, []) + patterns.get('en', []):
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 0:
                return match.strip()
    
    return "Not found"

def extract_year_advanced(text):
    """Advanced year extraction"""
    current_year = datetime.now().year
    
    # Look for years in various formats
    year_patterns = [
        r'Year\s*[:\-]\s*(\d{4})',
        r'वर्ष\s*[:\-]\s*(\d{4})',
        r'of\s+(\d{4})',
        r'दिनांक\s+\d{1,2}[-/]\d{1,2}[-/](\d{4})',
        r'(\d{4})\s*में',
        r'dated\s+\d{1,2}[-/]\d{1,2}[-/](\d{4})',
        r'year\s+(\d{4})',
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            if 1950 <= year <= current_year + 1:
                return str(year)
    
    # Look for any 4-digit number that could be a year
    year_matches = re.findall(r'\b(19[5-9]\d|20[0-9]{2})\b', text)
    valid_years = [int(y) for y in year_matches if 1950 <= int(y) <= current_year + 1]
    
    if valid_years:
        return str(max(valid_years))
    
    return "Not found"

def extract_acts_sections_advanced(text):
    """Advanced extraction of acts and sections"""
    # Improved patterns for Indian legal codes
    section_patterns = [
        r'\b(Sec\.|Section|S\.|धारा|उपधारा|कलम|U/S)\s*(\d+[A-Z]*(?:\s*[\(\)\d+\,\.\-andऔर及以及undve]+)*)',
        r'\b(Sections|Sections|धाराएं)\s+([\d+\,\.\-\sandऔर及以及undve]+)',
    ]
    
    act_patterns = [
        r'\b(IPC|CrPC|Indian Penal Code|Code of Criminal Procedure|भारतीय दंड संहिता|दंड प्रक्रिया संहिता|आईपीसी|सीआरपीसी)\s*(\,|\s|\(|\))',
        r'\b(SC\/ST Act|अधिनियम|Act|एक्ट)\s+([A-Za-z0-9\s\-]+)',
    ]
    
    sections = []
    acts = []
    
    # Extract sections
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            section_text = match.group(2).strip()
            # Clean and split sections
            if re.search(r'[,\sandऔर及以及undve]', section_text):
                split_sections = re.split(r'[,\sandऔर及以及undve]+', section_text)
                sections.extend([s.strip() for s in split_sections if s.strip().isdigit()])
            else:
                sections.append(section_text)
    
    # Extract acts
    for pattern in act_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            act_text = match.group(1).strip()
            acts.append(act_text)
    
    # Remove duplicates and clean
    sections = list(dict.fromkeys([s for s in sections if s]))
    acts = list(dict.fromkeys([a for a in acts if a]))
    
    return acts, sections

def extract_locations_advanced(text, ner_pipeline, language):
    """Advanced location extraction"""
    locations = []
    
    # Use NER
    if ner_pipeline:
        try:
            # Process in chunks to avoid memory issues
            chunks = [text[i:i+1000] for i in range(0, min(len(text), 5000), 1000)]
            for chunk in chunks:
                entities = ner_pipeline(chunk)
                for entity in entities:
                    if entity['entity_group'] in ['LOC', 'GPE'] and len(entity['word']) > 2:
                        locations.append(entity['word'])
        except:
            pass
    
    # Indian-specific location patterns
    location_patterns = [
        r'Police Station\s*[:\-]\s*([^\n,\.]+)',
        r'P\.?S\.?\s*[:\-]\s*([^\n,\.]+)',
        r'Station\s*[:\-]\s*([^\n,\.]+)',
        r'पुलिस स्टेशन\s*[:\-]\s*([^\n,\.]+)',
        r'थाना\s*[:\-]\s*([^\n,\.]+)',
        r'District\s*[:\-]\s*([^\n,\.]+)',
        r'जिला\s*[:\-]\s*([^\n,\.]+)',
        r'State\s*[:\-]\s*([^\n,\.]+)',
        r'राज्य\s*[:\-]\s*([^\n,\.]+)',
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        locations.extend([m.strip() for m in matches if m.strip()])
    
    return list(dict.fromkeys(locations))

def extract_names_advanced(text, ner_pipeline, language):
    """Advanced name extraction"""
    names = []
    
    # Use NER
    if ner_pipeline:
        try:
            chunks = [text[i:i+1000] for i in range(0, min(len(text), 5000), 1000)]
            for chunk in chunks:
                entities = ner_pipeline(chunk)
                for entity in entities:
                    if entity['entity_group'] == 'PER' and len(entity['word']) > 2:
                        names.append(entity['word'])
        except:
            pass
    
    # Name patterns for Indian names
    name_patterns = [
        r'Shri\s+([A-Za-z\s]+?)(?=\s|\,|\.|$)',
        r'Smt\.?\s+([A-Za-z\s]+?)(?=\s|\,|\.|$)',
        r'Mr\.?\s+([A-Za-z\s]+?)(?=\s|\,|\.|$)',
        r'Mrs\.?\s+([A-Za-z\s]+?)(?=\s|\,|\.|$)',
        r'Ms\.?\s+([A-Za-z\s]+?)(?=\s|\,|\.|$)',
        r'श्री\s+([^\n,\.]+?)(?=\s|\,|\.|$)',
        r'सौम्या\s+([^\n,\.]+?)(?=\s|\,|\.|$)',
        r'कुमार\s+([^\n,\.]+?)(?=\s|\,|\.|$)',
        r'Complainant\s*[:\-]\s*([^\n,\.]+)',
        r'Accused\s*[:\-]\s*([^\n,\.]+)',
        r'शिकायतकर्ता\s*[:\-]\s*([^\n,\.]+)',
        r'अभियुक्त\s*[:\-]\s*([^\n,\.]+)',
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        names.extend([m.strip() for m in matches if m.strip()])
    
    return list(dict.fromkeys(names))

def extract_with_qa_advanced(qa_pipeline, text, questions):
    """Advanced QA extraction with context optimization"""
    results = {}
    if not qa_pipeline or not text:
        return results
    
    # Find the most relevant context for each question
    for field, question in questions.items():
        try:
            # Try to find the best context window for this question
            context_window = text[:3000]  # Default to first 3000 chars
            
            # For specific fields, try to find better context
            if 'category' in field.lower():
                # Look for context around words like "category", "type", "offense"
                category_keywords = ['category', 'type', 'offense', 'offence', 'प्रकार', 'श्रेणी']
                for keyword in category_keywords:
                    if keyword in text.lower():
                        start = max(0, text.lower().find(keyword) - 200)
                        end = min(len(text), text.lower().find(keyword) + 500)
                        context_window = text[start:end]
                        break
            
            answer = qa_pipeline(question=question, context=context_window)
            if answer['score'] > 0.15:  # Lower threshold for legal documents
                results[field] = answer['answer']
            else:
                results[field] = "Not found"
        except:
            results[field] = "Error extracting"
    
    return results

def main():
    st.title("🔍 ULTIMATE FIR PII Extraction Tool")
    st.markdown("### Advanced PII Extraction for Indian FIR Documents with Maximum Accuracy")
    
    # Main file upload area
    uploaded_file = st.file_uploader("Upload FIR PDF Document", type="pdf", help="Upload a FIR document in PDF format")
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            extraction_mode = st.radio(
                "Extraction Mode",
                ["Aggressive", "Balanced", "Conservative"],
                help="Aggressive: More results but possible errors, Conservative: Fewer but more accurate results"
            )
        with col2:
            language_preference = st.selectbox(
                "Language Preference",
                ["Auto-Detect", "English", "Hindi", "Mixed"],
                help="Force specific language processing"
            )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing document with advanced techniques..."):
            # Extract text
            extracted_text = extract_text_advanced(uploaded_file)
            
            if extracted_text and len(extracted_text.strip()) > 50:
                st.session_state.extracted_text = extracted_text
                
                # Detect language
                detected_language = detect_language_advanced(extracted_text)
                if language_preference != "Auto-Detect":
                    detected_language = language_preference.lower()
                
                st.info(f"**Detected Language**: {detected_language.upper()} | **Text Length**: {len(extracted_text)} characters")
                
                if st.session_state.ocr_used:
                    st.success("📄 Used advanced OCR for text extraction")
                
                # Display text preview
                with st.expander("View Extracted Text Preview", expanded=False):
                    st.text_area("Text", extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text, height=200)
                
                # Load models
                with st.spinner("Loading advanced NLP models..."):
                    ner_pipeline = load_ner_model()
                    qa_pipeline = load_qa_model()
                    legal_phrases = load_legal_phrases()
                
                # Extract information
                pii_data = {}
                
                # Basic information
                pii_data['fir_no'] = extract_fir_number_advanced(extracted_text, detected_language)
                pii_data['year'] = extract_year_advanced(extracted_text)
                
                # Legal information
                acts, sections = extract_acts_sections_advanced(extracted_text)
                pii_data['under_acts'] = acts
                pii_data['under_sections'] = sections
                
                # Entity extraction
                pii_data['locations'] = extract_locations_advanced(extracted_text, ner_pipeline, detected_language)
                pii_data['names'] = extract_names_advanced(extracted_text, ner_pipeline, detected_language)
                
                # QA extraction for complex fields
                qa_questions = {
                    "revised_case_category": "What is the category or type of crime or case?",
                    "oparty": "Who is the accused or opposite party or complainant?",
                    "jurisdiction": "What is the jurisdiction or legal authority?",
                    "address": "What is the address or location mentioned?",
                    "jurisdiction_type": "What type of jurisdiction is this?"
                }
                
                qa_results = extract_with_qa_advanced(qa_pipeline, extracted_text, qa_questions)
                pii_data.update(qa_results)
                
                st.session_state.pii_data = pii_data
                
                # Display results in an organized manner
                st.success("✅ Extraction Complete!")
                
                # Create tabs for different types of information
                tab1, tab2, tab3, tab4 = st.tabs(["FIR Details", "Legal Information", "Parties Involved", "Raw Data"])
                
                with tab1:
                    st.subheader("FIR Basic Details")
                    
                    fir_details = pd.DataFrame({
                        'Field': ['FIR Number', 'Year', 'Jurisdiction', 'Jurisdiction Type'],
                        'Value': [
                            pii_data.get('fir_no', 'Not found'),
                            pii_data.get('year', 'Not found'),
                            pii_data.get('jurisdiction', 'Not found'),
                            pii_data.get('jurisdiction_type', 'Not found')
                        ]
                    })
                    
                    st.table(fir_details)
                
                with tab2:
                    st.subheader("Legal Information")
                    
                    # Display acts and sections
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Acts Under Which Case is Registered**")
                        if pii_data.get('under_acts'):
                            for act in pii_data['under_acts']:
                                st.info(f"• {act}")
                        else:
                            st.info("No acts found")
                    
                    with col2:
                        st.write("**Sections Under FIR**")
                        if pii_data.get('under_sections'):
                            for section in pii_data['under_sections']:
                                st.info(f"• {section}")
                        else:
                            st.info("No sections found")
                    
                    st.write("**Case Category**")
                    st.info(pii_data.get('revised_case_category', 'Not found'))
                
                with tab3:
                    st.subheader("Parties and Locations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Names Identified**")
                        if pii_data.get('names'):
                            for name in pii_data['names']:
                                st.success(f"• {name}")
                        else:
                            st.info("No names detected")
                        
                        st.write("**Opposite Party/Complainant**")
                        st.info(pii_data.get('oparty', 'Not found'))
                    
                    with col2:
                        st.write("**Locations Identified**")
                        if pii_data.get('locations'):
                            for location in pii_data['locations']:
                                st.success(f"• {location}")
                        else:
                            st.info("No locations detected")
                        
                        st.write("**Address**")
                        st.info(pii_data.get('address', 'Not found'))
                
                with tab4:
                    st.subheader("Raw Extracted Data")
                    st.json(pii_data)
                
                # Download options
                st.subheader("Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV download
                    csv_data = []
                    for key, value in pii_data.items():
                        if isinstance(value, list):
                            csv_data.append({'Field': key, 'Value': ', '.join(value)})
                        else:
                            csv_data.append({'Field': key, 'Value': value})
                    
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="fir_extraction_results.csv",
                        mime="text/csv",
                    )
                
                with col2:
                    # JSON download
                    json_str = json.dumps(pii_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Download as JSON",
                        data=json_str,
                        file_name="fir_extraction_results.json",
                        mime="application/json",
                    )
                
                # Confidence metrics
                st.info(f"**Extraction Confidence**: {'High' if pii_data.get('fir_no', 'Not found') != 'Not found' else 'Medium'} | "
                       f"**Entities Found**: {len(pii_data.get('names', [])) + len(pii_data.get('locations', []))}")
                
            else:
                st.error("❌ Could not extract sufficient text from the document. The PDF might be scanned or encrypted.")
    else:
        # Show welcome message and instructions
        st.markdown("""
        ## Welcome to the ULTIMATE FIR PII Extraction Tool
        
        This tool uses state-of-the-art AI and NLP techniques to extract Personal Identifiable Information 
        from Indian FIR documents with maximum accuracy.
        
        ### How to use:
        1. Upload a FIR document in PDF format
        2. The tool will automatically detect the language and extract information
        3. Review the extracted data in the organized tabs
        4. Download the results in CSV or JSON format
        
        ### Supported Features:
        - **Multi-language Support**: English, Hindi, and other Indian languages
        - **Advanced OCR**: Handles both digital and scanned PDFs
        - **Comprehensive Extraction**: FIR numbers, legal sections, names, addresses, and more
        - **High Accuracy**: Uses ensemble methods combining NER, pattern matching, and QA
        """)
        
        # Show sample output
        with st.expander("View Sample Output"):
            sample_data = {
                "fir_no": "123/2023",
                "year": "2023",
                "under_acts": ["IPC 1860", "SC/ST Act"],
                "under_sections": ["323", "506", "34"],
                "revised_case_category": "Assault and Criminal Intimidation",
                "oparty": "Rajesh Kumar",
                "names": ["Rajesh Kumar", "Sunita Devi", "Inspector Singh"],
                "address": "123 Main Road, Delhi",
                "jurisdiction": "District Court, South Delhi",
                "jurisdiction_type": "Local",
                "locations": ["Delhi", "South Delhi", "Police Station Saket"]
            }
            st.json(sample_data)

if __name__ == "__main__":
    main()
