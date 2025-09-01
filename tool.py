import streamlit as st
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import re
import pandas as pd
from datetime import datetime
import io
from typing import Dict, List, Tuple, Any
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Legal PDF PII Extractor",
    page_icon="📄",
    layout="wide"
)

class PIIExtractor:
    def __init__(self):
        # Comprehensive regex patterns for various PII types
        self.patterns = {
            'phone_numbers': [
                r'\b(?:\+91|91)?[-.\s]?[6789]\d{9}\b',  # Indian mobile numbers
                r'\b\d{4}[-.\s]?\d{3}[-.\s]?\d{4}\b',   # General phone format
                r'(?:मो\.|मोबाइल|Phone|mobile|फोन)\s*(?:नं\.|नंबर|No\.?)?\s*:?\s*(\d+)',
            ],
            'names': [
                r'(?:नाव|Name|NAAM)\s*[:।]\s*([A-Z\s]+(?:[A-Z][a-z]*\s*)*)',
                r'(?:Father|वडिलांचे|Husband|पतीचे)\s*(?:नाव|Name)\s*[:।]\s*([A-Z\s]+)',
                r'(?:Complainant|तक्रारदार|शिकायतकर्ता)\s*[:।]\s*([A-Z\s]+)',
            ],
            'addresses': [
                r'(?:पत्ता|Address|PATTA)\s*[:।]\s*([^,\n]+(?:,[^,\n]+)*)',
                r'(?:रा\.|रहिवासी|Resident)\s*[:।]?\s*([^,\n]+(?:,[^,\n]+)*)',
                r'(?:शहर|City|जिल्हा|District|State|राज्य)\s*[:।]\s*([^,\n]+)',
            ],
            'dates': [
                r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b',
                r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})\b',
                r'(?:दिनांक|Date)\s*[:।]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            ],
            'fir_numbers': [
                r'(?:FIR|F\.I\.R\.|एफ\.आई\.आर\.)\s*(?:No\.|नं\.?|क्रमांक)\s*[:।]?\s*(\d+)',
                r'गु\.र\.नं\.\s*(\d+[\/\-]\d+)',
                r'गुन्हा\s*नंबर\s*[:।]\s*(\d+)',
            ],
            'police_stations': [
                r'(?:P\.S\.|पोलीस\s*स्टेशन|ठाणे)\s*[:।]?\s*([^,\n]+)',
                r'(?:Police\s*Station)\s*[:।]?\s*([^,\n]+)',
            ],
            'ages': [
                r'(?:वय|Age|Years)\s*[:।]?\s*(\d+)\s*(?:वर्षे?|years?|yrs?)',
                r'(\d+)\s*(?:वर्षे?|years?|yrs?)',
            ],
            'aadhaar': [
                r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',
                r'(?:आधार|Aadhaar|UID)\s*(?:नं\.|No\.?)\s*[:।]?\s*(\d{4}[-.\s]?\d{4}[-.\s]?\d{4})',
            ],
            'pan': [
                r'\b[A-Z]{5}\d{4}[A-Z]\b',
                r'(?:PAN|पॅन)\s*(?:नं\.|No\.?)\s*[:।]?\s*([A-Z]{5}\d{4}[A-Z])',
            ],
            'amounts': [
                r'(?:रु\.|Rs\.?|रुपये)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:रु\.|Rs\.?|रुपये)',
                r'मूल्य\s*[:।]\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            ],
            'vehicle_numbers': [
                r'\b[A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{4}\b',
                r'(?:वाहन|Vehicle)\s*(?:नं\.|No\.?)\s*[:।]?\s*([A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{4})',
            ],
            'case_types': [
                r'(?:कलम|Section|धारा)\s*(\d+(?:\([^)]+\))?)',
                r'(?:IPC|भारतीय\s*दंड\s*संहिता|आयपीसी)\s*(\d+)',
            ]
        }
        
        # Common Hindi-English name indicators
        self.name_indicators = [
            'नाव', 'Name', 'NAAM', 'वडिलांचे', 'Father', 'पतीचे', 'Husband',
            'तक्रारदार', 'Complainant', 'आरोपी', 'Accused', 'संशयित', 'Suspect'
        ]
        
        # Location indicators
        self.location_indicators = [
            'पत्ता', 'Address', 'रहिवासी', 'Resident', 'शहर', 'City', 
            'जिल्हा', 'District', 'राज्य', 'State', 'गाव', 'Village',
            'तालुका', 'Tehsil', 'पिन', 'PIN'
        ]

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text using multiple methods for best results"""
        text = ""
        
        try:
            # Method 1: Using pdfplumber (best for mixed content)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}")
            
        # Method 2: Using PyMuPDF as fallback
        if not text.strip():
            try:
                pdf_file.seek(0)
                pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text += page.get_text() + "\n"
                pdf_document.close()
            except Exception as e:
                st.warning(f"PyMuPDF extraction failed: {e}")
                
        # Method 3: Using PyPDF2 as last resort
        if not text.strip():
            try:
                pdf_file.seek(0)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                st.warning(f"PyPDF2 extraction failed: {e}")
                
        return text

    def clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Hindi and English
        text = re.sub(r'[^\u0900-\u097F\u0020-\u007E\n]', ' ', text)
        return text.strip()

    def extract_pii_by_category(self, text: str) -> Dict[str, List[str]]:
        """Extract PII using pattern matching"""
        extracted_pii = {}
        
        for category, patterns in self.patterns.items():
            extracted_pii[category] = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Handle tuple matches (like date groups)
                    if isinstance(matches[0], tuple):
                        matches = ['/'.join(match) for match in matches]
                    extracted_pii[category].extend(matches)
            
            # Remove duplicates and clean
            extracted_pii[category] = list(set([
                match.strip() for match in extracted_pii[category] 
                if match and match.strip()
            ]))
        
        return extracted_pii

    def extract_contextual_pii(self, text: str) -> Dict[str, Any]:
        """Extract PII using contextual analysis"""
        lines = text.split('\n')
        contextual_pii = {
            'complainant_details': {},
            'accused_details': [],
            'case_details': {},
            'incident_details': {},
            'property_details': []
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Extract complainant details
            if any(indicator in line for indicator in ['Complainant', 'तक्रारदार']):
                name_match = re.search(r'(?:नाव|Name)\s*[:।]\s*([A-Z\s]+)', line, re.IGNORECASE)
                if name_match:
                    contextual_pii['complainant_details']['name'] = name_match.group(1).strip()
                    
            # Extract case numbers and dates
            if 'FIR' in line or 'एफ.आई.आर' in line:
                fir_match = re.search(r'(\d+)', line)
                if fir_match:
                    contextual_pii['case_details']['fir_number'] = fir_match.group(1)
                    
            # Extract property/amount details
            if any(word in line for word in ['रु.', 'Rs.', 'मूल्य', 'Value']):
                amount_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', line)
                if amount_match:
                    contextual_pii['property_details'].append({
                        'amount': amount_match.group(1),
                        'context': line[:100]
                    })
        
        return contextual_pii

    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data specific to FIR format"""
        structured_data = {}
        
        # Extract FIR header information
        fir_header_patterns = {
            'police_station': r'P\.S\.\s*\(पोलीस\s*ठाणे\)\s*[:।]\s*([^F]+?)(?=FIR|$)',
            'fir_number': r'FIR\s*No\.\s*[:।]\s*(\d+)',
            'district': r'District\s*\(जिल्हा\)\s*[:।]\s*([^Y]+?)(?=Year|$)',
            'year': r'Year\s*\(वर्ष\)\s*[:।]\s*(\d{4})',
            'date_time': r'Date\s*and\s*Time.*?(\d{1,2}/\d{1,2}/\d{4})\s*(\d{1,2}:\d{2})'
        }
        
        for key, pattern in fir_header_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                if key == 'date_time':
                    structured_data['date'] = match.group(1)
                    structured_data['time'] = match.group(2)
                else:
                    structured_data[key] = match.group(1).strip()
        
        # Extract accused persons details
        accused_section = re.search(
            r'Details\s*of\s*known.*?accused.*?:(.*?)(?=Particulars|$)', 
            text, re.IGNORECASE | re.DOTALL
        )
        
        if accused_section:
            accused_text = accused_section.group(1)
            # Extract individual accused details
            name_matches = re.findall(r'Name\s*\(नाव\)\s*([^A]+?)(?=Alias|Name|$)', accused_text)
            structured_data['accused_names'] = [name.strip() for name in name_matches if name.strip()]
        
        return structured_data

def main():
    st.title("🔍 Legal PDF PII Extractor")
    st.markdown("### Extract Personal Identifiable Information from Legal Documents (FIRs)")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF file", 
        type=['pdf'], 
        help="Upload legal documents like FIRs for PII extraction"
    )
    
    if uploaded_file is not None:
        # Initialize extractor
        extractor = PIIExtractor()
        
        with st.spinner("Processing PDF..."):
            # Extract text
            extracted_text = extractor.extract_text_from_pdf(uploaded_file)
            cleaned_text = extractor.clean_text(extracted_text)
            
        if not cleaned_text.strip():
            st.error("❌ Could not extract text from PDF. Please check if the file is valid.")
            return
            
        # Display extracted text preview
        with st.expander("📄 Extracted Text Preview"):
            st.text_area("Raw Text", cleaned_text[:2000] + "..." if len(cleaned_text) > 2000 else cleaned_text, height=200)
        
        # Extract PII
        with st.spinner("Extracting PII..."):
            pii_data = extractor.extract_pii_by_category(cleaned_text)
            contextual_pii = extractor.extract_contextual_pii(cleaned_text)
            structured_data = extractor.extract_structured_data(cleaned_text)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 PII Summary", "🔍 Detailed Analysis", "📊 Structured Data", "📥 Export"])
        
        with tab1:
            st.header("Extracted PII Summary")
            
            # Create summary cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Phone Numbers", len(pii_data.get('phone_numbers', [])))
                st.metric("Names Found", len(pii_data.get('names', [])))
                
            with col2:
                st.metric("Addresses", len(pii_data.get('addresses', [])))
                st.metric("Dates", len(pii_data.get('dates', [])))
                
            with col3:
                st.metric("FIR Numbers", len(pii_data.get('fir_numbers', [])))
                st.metric("Amounts", len(pii_data.get('amounts', [])))
            
            # Display key information
            if pii_data.get('names'):
                st.subheader("👤 Names")
                for name in pii_data['names'][:5]:  # Show top 5
                    st.write(f"• {name}")
                    
            if pii_data.get('phone_numbers'):
                st.subheader("📱 Phone Numbers")
                for phone in pii_data['phone_numbers']:
                    st.write(f"• {phone}")
                    
            if pii_data.get('addresses'):
                st.subheader("🏠 Addresses")
                for addr in pii_data['addresses'][:3]:  # Show top 3
                    st.write(f"• {addr}")
        
        with tab2:
            st.header("Detailed PII Analysis")
            
            for category, items in pii_data.items():
                if items:
                    st.subheader(f"{category.replace('_', ' ').title()}")
                    df = pd.DataFrame({'Extracted Data': items})
                    st.dataframe(df, use_container_width=True)
        
        with tab3:
            st.header("Structured Document Data")
            
            if structured_data:
                st.json(structured_data)
            
            if contextual_pii:
                st.subheader("Contextual Information")
                
                # Complainant details
                if contextual_pii.get('complainant_details'):
                    st.write("**Complainant Details:**")
                    st.json(contextual_pii['complainant_details'])
                
                # Case details
                if contextual_pii.get('case_details'):
                    st.write("**Case Details:**")
                    st.json(contextual_pii['case_details'])
                
                # Property details
                if contextual_pii.get('property_details'):
                    st.write("**Property/Financial Details:**")
                    for prop in contextual_pii['property_details']:
                        st.write(f"• Amount: {prop.get('amount', 'N/A')}")
                        st.write(f"  Context: {prop.get('context', 'N/A')[:100]}...")
        
        with tab4:
            st.header("Export Results")
            
            # Prepare export data
            export_data = {
                'extraction_timestamp': datetime.now().isoformat(),
                'file_name': uploaded_file.name,
                'pii_data': pii_data,
                'contextual_pii': contextual_pii,
                'structured_data': structured_data
            }
            
            # JSON export
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # CSV export for tabular data
            all_pii_items = []
            for category, items in pii_data.items():
                for item in items:
                    all_pii_items.append({
                        'Category': category,
                        'Value': item,
                        'File': uploaded_file.name
                    })
            
            if all_pii_items:
                df = pd.DataFrame(all_pii_items)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv,
                    file_name=f"pii_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Advanced options
        with st.sidebar:
            st.header("⚙️ Advanced Options")
            
            show_confidence = st.checkbox("Show Confidence Scores", value=False)
            filter_duplicates = st.checkbox("Filter Duplicates", value=True)
            min_length = st.slider("Minimum Text Length", 2, 20, 3)
            
            st.header("📊 Statistics")
            total_pii = sum(len(items) for items in pii_data.values())
            st.metric("Total PII Items", total_pii)
            st.metric("Text Length", len(cleaned_text))
            st.metric("Categories Found", len([k for k, v in pii_data.items() if v]))

if __name__ == "__main__":
    main()

# Additional utility functions for enhanced extraction

def enhance_hindi_extraction(text: str) -> str:
    """Enhance Hindi text extraction with specific preprocessing"""
    # Common Devanagari character corrections
    corrections = {
        'ɮ': 'व', 'ĵ': 'ज', 'ç': 'क', 'Ě': 'ष', 'ã': 'न',
        'Ĥ': 'प', 'ʜ': 'स', 'Ͻ': 'त', 'ǔ': 'ज', 'Ǖ': 'उ'
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def validate_extracted_data(pii_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Validate and clean extracted PII data"""
    validated_data = {}
    
    for category, items in pii_data.items():
        validated_items = []
        
        for item in items:
            # Validate phone numbers
            if category == 'phone_numbers':
                if re.match(r'^\d{10,12}$', re.sub(r'[\s\-\.]', '', item)):
                    validated_items.append(item)
            
            # Validate names (must contain at least one letter)
            elif category == 'names':
                if re.search(r'[A-Za-z\u0900-\u097F]', item) and len(item.strip()) > 2:
                    validated_items.append(item)
            
            # Validate dates
            elif category == 'dates':
                try:
                    # Try to parse the date
                    date_parts = re.findall(r'\d+', item)
                    if len(date_parts) >= 3:
                        validated_items.append(item)
                except:
                    pass
            
            else:
                if len(item.strip()) > 1:
                    validated_items.append(item)
        
        validated_data[category] = validated_items
    
    return validated_data

# Usage instructions
"""
USAGE INSTRUCTIONS:
==================

1. Install required packages:
   pip install streamlit PyPDF2 pdfplumber PyMuPDF pandas

2. Run the application:
   streamlit run pii_extractor.py

3. Upload your PDF file and view extracted PII in different tabs

4. Export results as JSON or CSV for further processing

FEATURES:
=========
- Multi-method PDF text extraction for maximum accuracy
- Hindi and English text processing
- Pattern-based PII extraction
- Contextual analysis for better accuracy
- Structured data extraction for FIR documents
- Data validation and cleaning
- Export functionality
- Real-time preview and statistics

SUPPORTED PII TYPES:
===================
- Names (complainant, accused, witnesses)
- Phone numbers (mobile and landline)
- Addresses (residential, incident location)
- Dates (incident, filing, birth dates)
- FIR numbers and case references
- Ages and personal details
- Financial amounts
- Vehicle numbers
- Legal sections and acts
- Aadhaar and PAN numbers
"""
