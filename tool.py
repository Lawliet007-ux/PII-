import streamlit as st
import pandas as pd
import re
import PyPDF2
import fitz  # PyMuPDF
from datetime import datetime
import json
import io
from collections import defaultdict
import spacy
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="Legal PII Extractor",
    page_icon="📄",
    layout="wide"
)

class PIIExtractor:
    def __init__(self):
        self.pii_patterns = {
            # Names (English and Devanagari patterns)
            'names': [
                r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b',  # English names
                r'(?:नाव|Name)[\s:]+([A-Za-z\s]+)',
                r'(?:वडिलांचे|Father)[\s:]+([A-Za-z\s]+)',
                r'(?:पिचे|Husband)[\s:]+([A-Za-z\s]+)',
            ],
            
            # Phone numbers
            'phone_numbers': [
                r'(?:मोबाइल|Mobile|Phone|फोन)[\s:]*(\d{10})',
                r'(?:मो\.|Mo\.)[\s:]*(\d{10})',
                r'\b\d{10}\b',
                r'\+91[-\s]?\d{10}',
            ],
            
            # Addresses
            'addresses': [
                r'(?:पता|Address|रा\.|Resi)[\s:]+([^,\n]+(?:,[^,\n]+)*)',
                r'(?:District|जिल्हा|ǔजिल्ला)[\s:]+([A-Za-z\s]+)',
                r'(?:State|राज्य)[\s:]+([A-Za-z\s]+)',
                r'(?:पुणे|Mumbai|Delhi|Kolkata|Chennai)[^,\n]*',
            ],
            
            # Dates
            'dates': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',
                r'(?:दिनांक|Date)[\s:]+(\d{1,2}/\d{1,2}/\d{4})',
                r'(?:जन्म|Birth)[\s:]+(\d{1,2}/\d{1,2}/\d{4})',
            ],
            
            # FIR Numbers
            'fir_numbers': [
                r'(?:FIR|एफ\.आई\.आर\.|गु\.र\.नं\.)[\s:]*(\d+/\d+)',
                r'\b\d+/\d{4}\b',
            ],
            
            # Police Station
            'police_stations': [
                r'(?:P\.S\.|पोलीस ठाणे|Police Station)[\s:]+([A-Za-z\s]+)',
                r'(?:ठाणे|थाना)[\s:]+([A-Za-z\s]+)',
            ],
            
            # ID Numbers
            'id_numbers': [
                r'(?:UID|आधार|Aadhar)[\s:]*(\d{12})',
                r'(?:PAN|पैन)[\s:]*([A-Z]{5}\d{4}[A-Z])',
                r'(?:Passport|पासपोर्ट)[\s:]*([A-Z]\d{7})',
                r'(?:Driving License|ड्रायव्हिंग)[\s:]*([A-Z]{2}\d{13})',
            ],
            
            # Vehicle Numbers
            'vehicle_numbers': [
                r'\b[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{2}[-\s]?\d{4}\b',
                r'(?:गाडी|Vehicle)[\s:]*([A-Z]{2}\d{2}[A-Z]{2}\d{4})',
            ],
            
            # Amount/Money
            'amounts': [
                r'(?:रु\.|Rs\.?|₹)[\s]*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'(?:मूल्य|Value)[\s:]*(?:Rs\.?|₹)?[\s]*(\d+(?:,\d+)*)',
            ],
            
            # Age
            'ages': [
                r'(?:वय|Age|उमर)[\s:]*(\d{1,3})(?:\s*(?:वर्ष|years?))?',
                r'\b\d{1,3}(?:\s*(?:वर्ष|years?))\b',
            ]
        }
        
        # Load spaCy model for NER (if available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            st.warning("SpaCy English model not found. Install with: python -m spacy download en_core_web_sm")
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF using PyMuPDF for better OCR support"""
        try:
            # Try PyMuPDF first (better for OCR PDFs)
            pdf_bytes = pdf_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
        
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s.,/:()-]', ' ', text)
        return text.strip()
    
    def extract_pii_regex(self, text):
        """Extract PII using regex patterns"""
        extracted_pii = defaultdict(list)
        
        for category, patterns in self.pii_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                        
                        match = match.strip()
                        if match and len(match) > 1:
                            # Additional validation
                            if self.validate_pii(category, match):
                                extracted_pii[category].append(match)
        
        return extracted_pii
    
    def validate_pii(self, category, value):
        """Validate extracted PII to reduce false positives"""
        value = value.strip()
        
        if category == 'phone_numbers':
            # Must be exactly 10 digits for Indian numbers
            digits_only = re.sub(r'\D', '', value)
            return len(digits_only) == 10 and digits_only.startswith(('6', '7', '8', '9'))
        
        elif category == 'names':
            # Names should have at least 2 characters and not be all uppercase common words
            if len(value) < 2:
                return False
            common_words = ['THE', 'AND', 'FOR', 'WITH', 'POLICE', 'STATION', 'CASE', 'NUMBER']
            return value.upper() not in common_words
        
        elif category == 'dates':
            # Basic date format validation
            return bool(re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', value))
        
        elif category == 'amounts':
            # Must contain digits
            return bool(re.search(r'\d', value))
        
        elif category == 'ages':
            # Age should be reasonable (1-120)
            age_num = re.search(r'\d+', value)
            if age_num:
                age = int(age_num.group())
                return 1 <= age <= 120
        
        return True
    
    def extract_pii_spacy(self, text):
        """Extract PII using spaCy NER"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        spacy_pii = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                spacy_pii['names'].append(ent.text)
            elif ent.label_ == "GPE":  # Geopolitical entity
                spacy_pii['locations'].append(ent.text)
            elif ent.label_ == "DATE":
                spacy_pii['dates'].append(ent.text)
            elif ent.label_ == "MONEY":
                spacy_pii['amounts'].append(ent.text)
        
        return spacy_pii
    
    def extract_contextual_pii(self, text):
        """Extract PII based on context and structure"""
        contextual_pii = defaultdict(list)
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for structured data patterns
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if any(word in key for word in ['name', 'नाव', 'naam']):
                    if value and len(value) > 2:
                        contextual_pii['names'].append(value)
                
                elif any(word in key for word in ['mobile', 'phone', 'मोबाइल', 'फोन']):
                    phone_match = re.search(r'\d{10}', value)
                    if phone_match:
                        contextual_pii['phone_numbers'].append(phone_match.group())
                
                elif any(word in key for word in ['address', 'पता', 'पत्ता']):
                    if value and len(value) > 5:
                        contextual_pii['addresses'].append(value)
        
        return contextual_pii
    
    def merge_pii_results(self, *pii_dicts):
        """Merge PII results from different extraction methods"""
        merged = defaultdict(set)
        
        for pii_dict in pii_dicts:
            for category, values in pii_dict.items():
                for value in values:
                    merged[category].add(value.strip())
        
        # Convert sets back to lists and remove duplicates
        final_pii = {}
        for category, values in merged.items():
            final_pii[category] = list(values)
        
        return final_pii
    
    def extract_all_pii(self, text):
        """Extract PII using all available methods"""
        # Clean the text first
        clean_text = self.clean_text(text)
        
        # Extract using different methods
        regex_pii = self.extract_pii_regex(clean_text)
        contextual_pii = self.extract_contextual_pii(clean_text)
        
        # Extract using spaCy if available
        spacy_pii = {}
        if self.nlp:
            spacy_pii = self.extract_pii_spacy(clean_text)
        
        # Merge all results
        final_pii = self.merge_pii_results(regex_pii, contextual_pii, spacy_pii)
        
        return final_pii

def main():
    st.title("🔍 Legal Document PII Extractor")
    st.markdown("**Extract Personal Identifiable Information from Legal PDFs (FIRs, Court Documents)**")
    
    # Initialize extractor
    extractor = PIIExtractor()
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Legal PDF Document",
        type=['pdf'],
        help="Upload FIR, Court documents, or other legal PDFs"
    )
    
    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Extract text
        with st.spinner("Extracting text from PDF..."):
            text = extractor.extract_text_from_pdf(uploaded_file)
        
        if text:
            # Display extracted text (preview)
            with st.expander("📄 Extracted Text Preview"):
                st.text_area("Text Content", text[:2000] + "..." if len(text) > 2000 else text, height=200)
            
            # Extract PII
            with st.spinner("Extracting PII..."):
                pii_data = extractor.extract_all_pii(text)
            
            # Display results
            st.header("🔍 Extracted PII Information")
            
            if pii_data:
                # Create tabs for different PII categories
                tabs = st.tabs(["📊 Summary", "👤 Personal Info", "📱 Contact Info", "🏛️ Legal Info", "💰 Financial Info", "📋 Export"])
                
                with tabs[0]:  # Summary
                    st.subheader("PII Summary")
                    
                    # Count statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_items = sum(len(values) for values in pii_data.values())
                        st.metric("Total PII Items", total_items)
                    
                    with col2:
                        st.metric("Categories Found", len(pii_data))
                    
                    with col3:
                        names_count = len(pii_data.get('names', []))
                        st.metric("Names Found", names_count)
                    
                    with col4:
                        phones_count = len(pii_data.get('phone_numbers', []))
                        st.metric("Phone Numbers", phones_count)
                    
                    # Category breakdown
                    if pii_data:
                        st.subheader("Category Breakdown")
                        category_df = pd.DataFrame([
                            {"Category": category.replace('_', ' ').title(), "Count": len(values)}
                            for category, values in pii_data.items()
                        ])
                        st.bar_chart(category_df.set_index('Category'))
                
                with tabs[1]:  # Personal Info
                    st.subheader("👤 Personal Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'names' in pii_data:
                            st.write("**Names:**")
                            for name in pii_data['names']:
                                st.write(f"• {name}")
                    
                    with col2:
                        if 'ages' in pii_data:
                            st.write("**Ages:**")
                            for age in pii_data['ages']:
                                st.write(f"• {age}")
                
                with tabs[2]:  # Contact Info
                    st.subheader("📱 Contact Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'phone_numbers' in pii_data:
                            st.write("**Phone Numbers:**")
                            for phone in pii_data['phone_numbers']:
                                st.write(f"• {phone}")
                    
                    with col2:
                        if 'addresses' in pii_data:
                            st.write("**Addresses:**")
                            for addr in pii_data['addresses']:
                                st.write(f"• {addr}")
                
                with tabs[3]:  # Legal Info
                    st.subheader("🏛️ Legal Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'fir_numbers' in pii_data:
                            st.write("**FIR Numbers:**")
                            for fir in pii_data['fir_numbers']:
                                st.write(f"• {fir}")
                        
                        if 'police_stations' in pii_data:
                            st.write("**Police Stations:**")
                            for ps in pii_data['police_stations']:
                                st.write(f"• {ps}")
                    
                    with col2:
                        if 'dates' in pii_data:
                            st.write("**Dates:**")
                            for date in pii_data['dates']:
                                st.write(f"• {date}")
                        
                        if 'vehicle_numbers' in pii_data:
                            st.write("**Vehicle Numbers:**")
                            for vehicle in pii_data['vehicle_numbers']:
                                st.write(f"• {vehicle}")
                
                with tabs[4]:  # Financial Info
                    st.subheader("💰 Financial Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'amounts' in pii_data:
                            st.write("**Amounts/Money:**")
                            for amount in pii_data['amounts']:
                                st.write(f"• {amount}")
                    
                    with col2:
                        if 'id_numbers' in pii_data:
                            st.write("**ID Numbers:**")
                            for id_num in pii_data['id_numbers']:
                                st.write(f"• {id_num}")
                
                with tabs[5]:  # Export
                    st.subheader("📋 Export Data")
                    
                    # Prepare data for export
                    export_data = []
                    for category, values in pii_data.items():
                        for value in values:
                            export_data.append({
                                "Category": category.replace('_', ' ').title(),
                                "Value": value,
                                "Document": uploaded_file.name,
                                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    
                    if export_data:
                        df = pd.DataFrame(export_data)
                        st.dataframe(df)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"pii_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON download
                            json_data = json.dumps(pii_data, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_data,
                                file_name=f"pii_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            else:
                st.warning("⚠️ No PII found in the document. The document might be:")
                st.write("- Heavily corrupted or poorly scanned")
                st.write("- Not in the expected format")
                st.write("- Contain only images without extractable text")
        else:
            st.error("❌ Could not extract text from the PDF. Please check if the file is valid.")
    
    # Instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown("""
        ### How to use this tool:
        
        1. **Upload a PDF**: Click on the file uploader and select your legal document (FIR, court documents, etc.)
        
        2. **Review Extracted Text**: Check the text preview to ensure proper extraction
        
        3. **View PII Results**: Navigate through different tabs to see categorized PII information
        
        4. **Export Data**: Download the extracted PII as CSV or JSON for further processing
        
        ### Supported PII Types:
        - **Names**: Person names in English and Hindi
        - **Phone Numbers**: 10-digit Indian mobile numbers
        - **Addresses**: Complete addresses with districts and states
        - **Dates**: Various date formats
        - **FIR Numbers**: Case numbers and reference numbers
        - **Police Stations**: Station names and locations
        - **ID Numbers**: Aadhaar, PAN, Passport, Driving License
        - **Vehicle Numbers**: Indian vehicle registration numbers
        - **Financial Info**: Amounts, money values
        - **Ages**: Person ages
        
        ### Features:
        - ✅ Multilingual support (Hindi, English)
        - ✅ OCR-processed document support
        - ✅ False positive reduction
        - ✅ Multiple extraction methods
        - ✅ Export capabilities
        - ✅ Real-time processing
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Legal PII Extractor** - Built for accurate and efficient PII extraction from legal documents")

if __name__ == "__main__":
    main()
