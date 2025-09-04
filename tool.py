import streamlit as st
import pdfplumber
import pandas as pd
import re
import spacy
from spacy import displacy
from collections import defaultdict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import numpy as np
from langdetect import detect
import easyocr

# Set page configuration
st.set_page_config(
    page_title="FIR PII Extraction Tool",
    page_icon="📄",
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

# Load models with caching
@st.cache_resource
def load_ner_model():
    """Load a multilingual NER model optimized for Indian languages"""
    try:
        return pipeline("ner", model="Davlan/xlm-roberta-large-ner-hrl", aggregation_strategy="average")
    except:
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource
def load_legal_ner_model():
    """Load a custom model for legal document parsing"""
    try:
        # This would ideally be a custom-trained model on Indian FIR documents
        tokenizer = AutoTokenizer.from_pretrained("legal-bert-base-uncased")
        model = AutoModelForTokenClassification.from_pretrained("legal-bert-base-uncased")
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")
    except:
        return None

@st.cache_resource
def load_qa_model():
    """Load a question answering model optimized for legal documents"""
    try:
        return pipeline("question-answering", model="deepset/roberta-base-squad2")
    except:
        return None

@st.cache_resource
def load_spacy_model():
    """Load spaCy model for English"""
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

@st.cache_resource
def load_easyocr_reader():
    """Load EasyOCR reader for multilingual text extraction"""
    return easyocr.Reader(['en', 'hi'])  # English and Hindi

def extract_text_with_ocr(pdf_file):
    """Extract text from PDF using OCR for scanned documents"""
    try:
        images = convert_from_bytes(pdf_file.read())
        text = ""
        reader = load_easyocr_reader()
        
        for image in images:
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=0)
            text += " ".join(results) + "\n"
            
        return text
    except Exception as e:
        st.error(f"OCR extraction failed: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using multiple methods"""
    text = ""
    
    # First try standard text extraction
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # If little text was extracted, try OCR
        if len(text.strip()) < 100:
            st.info("Text extraction yielded little content, trying OCR...")
            pdf_file.seek(0)  # Reset file pointer
            ocr_text = extract_text_with_ocr(pdf_file)
            if ocr_text and len(ocr_text) > 50:
                text = ocr_text
                st.session_state.ocr_used = True
                
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
        
    return text

def detect_language(text):
    """Detect the language of the text"""
    try:
        return detect(text)
    except:
        return "en"  # Default to English

def extract_fir_number(text):
    """Extract FIR number using pattern matching optimized for Indian FIR formats"""
    patterns = [
        r'FIR\s*No[\.\:\s]*([A-Za-z0-9\/\-]+)',
        r'F\.I\.R\.\s*No[\.\:\s]*([A-Za-z0-9\/\-]+)',
        r'First Information Report\s*No[\.\:\s]*([A-Za-z0-9\/\-]+)',
        r'Registration No[\.\:\s]*([A-Za-z0-9\/\-]+)',
        r'रिपोर्ट संख्या[\.\:\s]*([A-Za-z0-9\/\-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Not found"

def extract_year(text):
    """Extract year from the FIR"""
    year_patterns = [
        r'Year\s*[:\-]\s*(\d{4})',
        r'वर्ष\s*[:\-]\s*(\d{4})',
        r'of\s+(\d{4})',
        r'दिनांक\s+\d{1,2}[-/]\d{1,2}[-/](\d{4})',
        r'(\d{4})\s*में',
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: look for any 4-digit number between 1950 and current year + 1
    current_year = pd.Timestamp.now().year
    year_matches = re.findall(r'\b(19[5-9]\d|20[0-9]{2})\b', text)
    if year_matches:
        return max(year_matches)  # Return the most recent year
    
    return "Not found"

def extract_acts_sections(text):
    """Extract acts and sections mentioned in the FIR"""
    # Indian legal sections pattern
    section_pattern = r'\b(Sec\.|Section|S\.|धारा|उपधारा)\s*(\d+[A-Z]*(?:\s*[\(\)\d+\,\.\-andऔर]+)*)'
    act_pattern = r'\b(IPC|CrPC|Indian Penal Code|Code of Criminal Procedure|भारतीय दंड संहिता|दंड प्रक्रिया संहिता)\s*(\d{4})?'
    
    sections = []
    acts = []
    
    # Find sections
    section_matches = re.finditer(section_pattern, text, re.IGNORECASE)
    for match in section_matches:
        sections.append(match.group(2).strip())
    
    # Find acts
    act_matches = re.finditer(act_pattern, text, re.IGNORECASE)
    for match in act_matches:
        act_name = match.group(1)
        if match.group(2):  # If year is mentioned
            act_name += f" {match.group(2)}"
        acts.append(act_name)
    
    # Remove duplicates
    sections = list(dict.fromkeys(sections))
    acts = list(dict.fromkeys(acts))
    
    return acts, sections

def extract_locations(text, ner_pipeline):
    """Extract locations using NER and pattern matching"""
    locations = []
    
    # Use NER to find locations
    if ner_pipeline:
        try:
            entities = ner_pipeline(text[:1000])  # Process first 1000 chars to avoid long processing
            for entity in entities:
                if entity['entity_group'] == 'LOC':
                    locations.append(entity['word'])
        except:
            pass
    
    # Additional pattern matching for Indian police stations and districts
    ps_patterns = [
        r'Police Station\s*[:\-]\s*([^\n,]+)',
        r'P\.S\.\s*[:\-]\s*([^\n,]+)',
        r'thana\s*[:\-]\s*([^\n,]+)',
        r'पुलिस स्टेशन\s*[:\-]\s*([^\n,]+)',
        r'थाना\s*[:\-]\s*([^\n,]+)',
    ]
    
    for pattern in ps_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        locations.extend(matches)
    
    return list(dict.fromkeys(locations))

def extract_names(text, ner_pipeline):
    """Extract names using NER and pattern matching"""
    names = []
    
    # Use NER to find person names
    if ner_pipeline:
        try:
            entities = ner_pipeline(text[:1000])  # Process first 1000 chars
            for entity in entities:
                if entity['entity_group'] == 'PER':
                    names.append(entity['word'])
        except:
            pass
    
    # Pattern matching for common Indian name prefixes
    name_patterns = [
        r'Shri\s+([A-Za-z\s]+)',
        r'Smt\.\s+([A-Za-z\s]+)',
        r'Mr\.\s+([A-Za-z\s]+)',
        r'Mrs\.\s+([A-Za-z\s]+)',
        r'श्री\s+([^\n,]+)',
        r'सौम्या\s+([^\n,]+)',
        r'कुमार\s+([^\n,]+)',
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        names.extend(matches)
    
    return list(dict.fromkeys(names))

def extract_with_qa(qa_pipeline, text, questions):
    """Extract information using question answering"""
    results = {}
    if not qa_pipeline:
        return results
    
    for field, question in questions.items():
        try:
            answer = qa_pipeline(question=question, context=text[:2000])  # Limit context length
            if answer['score'] > 0.2:  # Slightly higher threshold for confidence
                results[field] = answer['answer']
            else:
                results[field] = "Not found"
        except:
            results[field] = "Error extracting"
    
    return results

def main():
    st.title("📄 Advanced FIR PII Extraction Tool")
    st.markdown("Extract Personal Identifiable Information from FIR documents in multiple Indian languages")
    
    # Main file upload area
    st.header("Upload FIR Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="visible")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        extraction_method = st.radio(
            "Extraction Method",
            ["Advanced NER", "Pattern Matching", "Combined Approach"],
            help="Choose the method for information extraction"
        )
        
        st.header("About")
        st.info("""
        This tool uses advanced NLP techniques to extract PII from FIR documents in multiple Indian languages.
        Supports both text-based and scanned PDFs using OCR.
        """)
    
    # Main content area
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text and len(extracted_text.strip()) > 50:
                st.session_state.extracted_text = extracted_text
                
                # Detect language
                language = detect_language(extracted_text)
                st.info(f"Detected language: {language}")
                
                if st.session_state.ocr_used:
                    st.success("Used OCR to extract text from scanned document")
                
                # Display extracted text
                with st.expander("View Extracted Text"):
                    st.text_area("Text", extracted_text, height=300)
                
                # Load models
                with st.spinner("Loading NLP models..."):
                    ner_pipeline = load_ner_model()
                    qa_pipeline = load_qa_model()
                
                # Extract information based on selected method
                pii_data = {}
                
                # Always extract these with specialized functions
                pii_data['fir_no'] = extract_fir_number(extracted_text)
                pii_data['year'] = extract_year(extracted_text)
                acts, sections = extract_acts_sections(extracted_text)
                pii_data['under_acts'] = acts
                pii_data['under_sections'] = sections
                
                if extraction_method in ["Advanced NER", "Combined Approach"]:
                    with st.spinner("Extracting PII with Advanced NER..."):
                        pii_data['locations'] = extract_locations(extracted_text, ner_pipeline)
                        pii_data['names'] = extract_names(extracted_text, ner_pipeline)
                
                if extraction_method in ["Pattern Matching", "Combined Approach"]:
                    with st.spinner("Extracting information with pattern matching..."):
                        # Additional pattern-based extraction
                        pass
                
                if extraction_method in ["Combined Approach"] and qa_pipeline:
                    with st.spinner("Extracting structured information with QA..."):
                        questions = {
                            "revised_case_category": "What is the case category or type of crime?",
                            "oparty": "Who is the accused or opposite party?",
                            "jurisdiction": "What is the jurisdiction of this case?",
                        }
                        qa_results = extract_with_qa(qa_pipeline, extracted_text, questions)
                        pii_data.update(qa_results)
                
                st.session_state.pii_data = pii_data
                
                # Display results
                st.subheader("Extracted Information")
                
                if st.session_state.pii_data:
                    # Create a structured display of information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**FIR Details**")
                        fir_details_data = {
                            'FIR Number': pii_data.get('fir_no', 'Not found'),
                            'Year': pii_data.get('year', 'Not found'),
                            'Acts': ', '.join(pii_data.get('under_acts', [])),
                            'Sections': ', '.join(pii_data.get('under_sections', [])),
                            'Case Category': pii_data.get('revised_case_category', 'Not found'),
                            'Jurisdiction': pii_data.get('jurisdiction', 'Not found'),
                        }
                        
                        for key, value in fir_details_data.items():
                            st.info(f"**{key}**: {value}")
                    
                    with col2:
                        st.write("**Parties Involved**")
                        if pii_data.get('names'):
                            st.info("**Names Found**: " + ", ".join(pii_data['names']))
                        else:
                            st.info("**Names Found**: Not detected")
                            
                        st.info("**Opposite Party**: " + pii_data.get('oparty', 'Not found'))
                        
                        if pii_data.get('locations'):
                            st.info("**Locations**: " + ", ".join(pii_data['locations']))
                        else:
                            st.info("**Locations**: Not detected")
                    
                    # Download button
                    csv_data = {
                        'Field': list(fir_details_data.keys()) + ['Names', 'Opposite Party', 'Locations'],
                        'Value': list(fir_details_data.values()) + [
                            ", ".join(pii_data.get('names', [])),
                            pii_data.get('oparty', 'Not found'),
                            ", ".join(pii_data.get('locations', []))
                        ]
                    }
                    
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="Download extracted data as CSV",
                        data=csv,
                        file_name="extracted_pii.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No PII could be extracted from the document.")
            else:
                st.error("Could not extract sufficient text from the uploaded PDF.")
    else:
        # Show instructions when no file is uploaded
        st.info("Please upload a FIR document in PDF format to begin analysis.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Supported Languages")
            st.markdown("""
            - English
            - Hindi
            - Other Indian languages
            """)
        
        with col2:
            st.subheader("Extracted Information")
            st.markdown("""
            - FIR Number and Year
            - Legal Sections and Acts
            - Names (Complainant/Accused)
            - Locations and Addresses
            - Case Categories
            - Jurisdiction Information
            """)
        
        with col3:
            st.subheader("Advanced Technology")
            st.markdown("""
            - Multilingual Transformer Models
            - Custom Pattern Matching
            - OCR for Scanned Documents
            - Question Answering Models
            - Hybrid Extraction Approach
            """)

if __name__ == "__main__":
    main()
