import streamlit as st
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
import re
import json
from typing import Dict, List, Any
import spacy
from spacy import displacy
import io
import base64
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class LegalPDFExtractor:
    def __init__(self):
        self.setup_patterns()
        
    def setup_patterns(self):
        """Setup regex patterns for PII extraction"""
        # Year patterns
        self.year_patterns = [
            r'\b(19|20)\d{2}\b',  # Standard year format
            r'वर्ष\s*(\d{4})',  # Hindi year
            r'साल\s*(\d{4})',   # Hindi year alternative
        ]
        
        # State name patterns (Indian states in English and Hindi)
        self.state_patterns = [
            r'(?:State|राज्य|प्रदेश)\s*:?\s*([A-Za-z\s]+)',
            r'(?:उत्तर प्रदेश|यूपी|UP|Uttar Pradesh)',
            r'(?:महाराष्ट्र|Maharashtra)',
            r'(?:बिहार|Bihar)',
            r'(?:पश्चिम बंगाल|West Bengal)',
            r'(?:तमिलनाडु|Tamil Nadu)',
            r'(?:कर्नाटक|Karnataka)',
            r'(?:गुजरात|Gujarat)',
            r'(?:राजस्थान|Rajasthan)',
            r'(?:ओडिशा|Odisha)',
            r'(?:तेलंगाना|Telangana)',
            r'(?:केरल|Kerala)',
            r'(?:असम|Assam)',
            r'(?:पंजाब|Punjab)',
            r'(?:हरियाणा|Haryana)',
            r'(?:छत्तीसगढ़|Chhattisgarh)',
            r'(?:झारखंड|Jharkhand)',
            r'(?:हिमाचल प्रदेश|Himachal Pradesh)',
            r'(?:उत्तराखंड|Uttarakhand)',
            r'(?:गोवा|Goa)',
            r'(?:मणिपुर|Manipur)',
            r'(?:मेघालय|Meghalaya)',
            r'(?:त्रिपुरा|Tripura)',
            r'(?:मिजोरम|Mizoram)',
            r'(?:अरुणाचल प्रदेश|Arunachal Pradesh)',
            r'(?:नागालैंड|Nagaland)',
            r'(?:सिक्किम|Sikkim)'
        ]
        
        # District name patterns
        self.district_patterns = [
            r'(?:District|जिला|जिल्हा)\s*:?\s*([A-Za-z\s]+)',
            r'(?:Dist|जिला)\.\s*([A-Za-z\s]+)',
            r'जिला\s*([A-Za-z\s]+)',
        ]
        
        # Police station patterns
        self.police_station_patterns = [
            r'(?:Police Station|थाना|पुलिस स्टेशन|PS)\s*:?\s*([A-Za-z\s]+)',
            r'(?:P\.S\.|PS|थाना)\s*:?\s*([A-Za-z\s]+)',
            r'थाना\s*([A-Za-z\s]+)',
        ]
        
        # Under Acts patterns
        self.under_acts_patterns = [
            r'(?:Under Act|अधिनियम|Act)\s*:?\s*(.*?)(?=\n|Section|\d+)',
            r'(?:IPC|भा\.द\.सं\.)\s*(\d{4})',
            r'(?:Indian Penal Code|भारतीय दंड संहिता)\s*(\d{4})',
            r'(?:Act|अधिनियम)\s*(\d{4})',
        ]
        
        # Under Sections patterns
        self.under_sections_patterns = [
            r'(?:Section|धारा|सेक्शन)\s*:?\s*([\d,\s/]+)',
            r'(?:U/S|धारा)\s*([\d,\s/]+)',
            r'धारा\s*([\d,\s/]+)',
            r'(?:Sec|Section)\.\s*([\d,\s/]+)',
        ]
        
        # Case category patterns
        self.case_category_patterns = [
            r'(?:FIR|प्राथमिकी|First Information Report)',
            r'(?:Case|मामला|केस)\s*(?:Type|Category|श्रेणी)\s*:?\s*([A-Za-z\s]+)',
            r'(?:Offence|अपराध|Crime|जुर्म)\s*:?\s*([A-Za-z\s]+)',
        ]
        
        # Name patterns (more flexible)
        self.name_patterns = [
            r'(?:Name|नाम|व्यक्ति)\s*:?\s*([A-Za-z\s]+)',
            r'(?:Accused|आरोपी|अभियुक्त)\s*:?\s*([A-Za-z\s]+)',
            r'(?:Complainant|शिकायतकर्ता|फरियादी)\s*:?\s*([A-Za-z\s]+)',
            r'(?:S/o|D/o|W/o|पुत्र|पुत्री|पत्नी)\s*([A-Za-z\s]+)',
        ]
        
        # Address patterns
        self.address_patterns = [
            r'(?:Address|पता|निवास)\s*:?\s*(.*?)(?=\n|Phone|Mobile)',
            r'(?:Village|गांव|ग्राम)\s*:?\s*([A-Za-z\s]+)',
            r'(?:Tehsil|तहसील)\s*:?\s*([A-Za-z\s]+)',
            r'(?:Block|ब्लॉक)\s*:?\s*([A-Za-z\s]+)',
        ]
        
        # Jurisdiction patterns
        self.jurisdiction_patterns = [
            r'(?:Jurisdiction|क्षेत्राधिकार)\s*:?\s*([A-Za-z\s_]+)',
            r'(?:PAN_INDIA|ALL_INDIA|NATIONAL)',
            r'(?:STATE|राज्य)',
            r'(?:DISTRICT|जिला)',
            r'(?:LOCAL|स्थानीय)',
        ]

    def extract_text_multiple_methods(self, pdf_file) -> str:
        """Extract text using multiple methods for better accuracy"""
        extracted_text = ""
        
        # Method 1: PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pypdf2_text = ""
            for page in pdf_reader.pages:
                pypdf2_text += page.extract_text() + "\n"
            extracted_text += pypdf2_text
        except Exception as e:
            st.warning(f"PyPDF2 extraction failed: {str(e)}")
        
        # Reset file pointer
        pdf_file.seek(0)
        
        # Method 2: pdfplumber (best for tables and complex layouts)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                pdfplumber_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdfplumber_text += page_text + "\n"
                extracted_text += pdfplumber_text
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {str(e)}")
        
        # Reset file pointer
        pdf_file.seek(0)
        
        # Method 3: PyMuPDF (best for OCR-like extraction)
        try:
            pdf_bytes = pdf_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            pymupdf_text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                pymupdf_text += page.get_text() + "\n"
            pdf_document.close()
            extracted_text += pymupdf_text
        except Exception as e:
            st.warning(f"PyMuPDF extraction failed: {str(e)}")
        
        return extracted_text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,;:()\-/]', ' ', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_using_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Extract information using regex patterns"""
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                results.extend(matches)
        return list(set(results))  # Remove duplicates

    def extract_year(self, text: str) -> List[str]:
        """Extract years from text"""
        years = []
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        # Filter valid years (1900-2030)
        valid_years = [year for year in years if 1900 <= int(year) <= 2030]
        return list(set(map(str, valid_years)))

    def extract_state_name(self, text: str) -> List[str]:
        """Extract state names from text"""
        states = []
        for pattern in self.state_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                states.extend([match.strip() for match in matches if isinstance(match, str)])
        return list(set(states))

    def extract_sections(self, text: str) -> List[str]:
        """Extract sections with better parsing"""
        sections = []
        for pattern in self.under_sections_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean and split sections
                clean_sections = re.split(r'[,/\s]+', match.strip())
                sections.extend([s.strip() for s in clean_sections if s.strip().isdigit()])
        return list(set(sections))

    def normalize_case_category(self, categories: List[str]) -> str:
        """Normalize case categories to standard format"""
        category_mapping = {
            'fir': 'First Information Report',
            'प्राथमिकी': 'First Information Report',
            'complaint': 'Complaint Case',
            'theft': 'Theft Case',
            'assault': 'Assault Case',
            'fraud': 'Fraud Case',
            'domestic violence': 'Domestic Violence Case',
        }
        
        for category in categories:
            for key, value in category_mapping.items():
                if key in category.lower():
                    return value
        
        return categories[0] if categories else "General Case"

    def determine_jurisdiction_type(self, text: str) -> str:
        """Determine jurisdiction type from text"""
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['pan india', 'all india', 'national']):
            return "PAN_INDIA"
        elif any(keyword in text_lower for keyword in ['state', 'राज्य']):
            return "STATE"
        elif any(keyword in text_lower for keyword in ['district', 'जिला']):
            return "DISTRICT"
        else:
            return "LOCAL"

    def extract_names(self, text: str) -> List[str]:
        """Extract names with better filtering"""
        names = []
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean name
                clean_name = re.sub(r'[^\w\s]', ' ', match).strip()
                # Filter out common non-names
                if (len(clean_name.split()) >= 1 and 
                    not any(word in clean_name.lower() for word in ['unknown', 'nil', 'na', 'not', 'available'])):
                    names.append(clean_name.title())
        return list(set(names))

    def extract_all_pii(self, text: str) -> Dict[str, Any]:
        """Extract all PII information from text"""
        clean_text = self.clean_text(text)
        
        # Extract each type of information
        pii_data = {
            'year': self.extract_year(clean_text),
            'state_name': self.extract_state_name(clean_text),
            'dist_name': self.extract_using_patterns(clean_text, self.district_patterns),
            'police_station': self.extract_using_patterns(clean_text, self.police_station_patterns),
            'under_acts': self.extract_using_patterns(clean_text, self.under_acts_patterns),
            'under_sections': self.extract_sections(clean_text),
            'case_categories': self.extract_using_patterns(clean_text, self.case_category_patterns),
            'names': self.extract_names(clean_text),
            'addresses': self.extract_using_patterns(clean_text, self.address_patterns),
            'jurisdiction': self.extract_using_patterns(clean_text, self.jurisdiction_patterns),
        }
        
        # Process and clean the data
        processed_pii = {
            'year': pii_data['year'],
            'state_name': [name.title() for name in pii_data['state_name']],
            'dist_name': [name.title() for name in pii_data['dist_name']],
            'police_station': [name.title() for name in pii_data['police_station']],
            'under_acts': pii_data['under_acts'],
            'under_sections': ', '.join(pii_data['under_sections']),
            'revised_case_category': self.normalize_case_category(pii_data['case_categories']),
            'oparty': pii_data['names'],
            'name': pii_data['names'],
            'address': pii_data['addresses'],
            'jurisdiction': pii_data['jurisdiction'],
            'jurisdiction_type': self.determine_jurisdiction_type(clean_text)
        }
        
        return processed_pii

def main():
    st.set_page_config(
        page_title="Legal PDF PII Extractor",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚖️ Legal PDF PII Extraction Tool")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📋 Instructions")
    st.sidebar.markdown("""
    **Supported Languages:** Hindi & English
    
    **Extracted Information:**
    - Year
    - State Name
    - District Name
    - Police Station
    - Under Acts
    - Under Sections
    - Case Category
    - Names (Accused/Complainant)
    - Address
    - Jurisdiction Information
    """)
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        st.session_state.extractor = LegalPDFExtractor()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Legal PDF Files (FIR documents)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files containing legal documents (mainly FIRs)"
    )
    
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
        
        # Process button
        if st.button("🔍 Extract PII Information", type="primary"):
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                try:
                    # Extract text
                    extracted_text = st.session_state.extractor.extract_text_multiple_methods(uploaded_file)
                    
                    # Extract PII
                    pii_data = st.session_state.extractor.extract_all_pii(extracted_text)
                    
                    # Add filename to results
                    pii_data['filename'] = uploaded_file.name
                    results.append(pii_data)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            if results:
                st.markdown("---")
                st.header("📊 Extraction Results")
                
                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Processed", len(results))
                with col2:
                    total_entities = sum(len(str(v).split(',')) for result in results for v in result.values() if isinstance(v, (str, list)))
                    st.metric("Total Entities Extracted", total_entities)
                with col3:
                    success_rate = (len(results) / len(uploaded_files)) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Detailed results
                for idx, result in enumerate(results):
                    st.subheader(f"📄 {result['filename']}")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Information:**")
                        st.write(f"**Year:** {', '.join(result['year']) if result['year'] else 'Not found'}")
                        st.write(f"**State:** {', '.join(result['state_name']) if result['state_name'] else 'Not found'}")
                        st.write(f"**District:** {', '.join(result['dist_name']) if result['dist_name'] else 'Not found'}")
                        st.write(f"**Police Station:** {', '.join(result['police_station']) if result['police_station'] else 'Not found'}")
                        st.write(f"**Case Category:** {result['revised_case_category']}")
                    
                    with col2:
                        st.write("**Legal Information:**")
                        st.write(f"**Under Acts:** {', '.join(result['under_acts']) if result['under_acts'] else 'Not found'}")
                        st.write(f"**Under Sections:** {result['under_sections'] if result['under_sections'] else 'Not found'}")
                        st.write(f"**Jurisdiction Type:** {result['jurisdiction_type']}")
                        st.write(f"**Names:** {', '.join(result['name']) if result['name'] else 'Not found'}")
                        st.write(f"**Address:** {', '.join(result['address']) if result['address'] else 'Not found'}")
                    
                    st.markdown("---")
                
                # Convert to DataFrame for export
                df_results = pd.DataFrame(results)
                
                # Export options
                st.header("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV export
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv,
                        file_name=f"pii_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # JSON export
                    json_data = json.dumps(results, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📋 Download as JSON",
                        data=json_data,
                        file_name=f"pii_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Display DataFrame
                st.header("📊 Tabular View")
                st.dataframe(df_results, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Legal PDF PII Extractor v1.0 | Supports Hindi & English | Optimized for FIR Documents</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
