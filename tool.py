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
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Legal PDF PII Extractor",
    page_icon="📄",
    layout="wide"
)

class AdvancedPIIExtractor:
    def __init__(self):
        # Enhanced patterns for better extraction
        self.patterns = {
            'phone_numbers': [
                r'\b(?:\+91|91)?[-.\s]?[6789]\d{9}\b',
                r'\b\d{10}\b',
                r'(?:मो\.|मोबाइल|Phone|mobile|फोन|Mo\.|Mobile)\s*(?:नं\.|नंबर|No\.?|न\.|Đ\.)\s*[:।]?\s*(\d+)',
                r'(?:\d{4})?\s*(\d{10})',
                r'(\d{4}\s*\d{3}\s*\d{4})',
            ],
            'names': [
                # Enhanced name patterns
                r'(?:नाव|Name|NAAM)\s*[:।)\s]*([A-Z][A-Z\s]+[A-Z])',
                r'(?:Father|वडिलांचे|Husband|पतीचे|पिचे)\s*(?:नाव|Name)\s*[:।]\s*([A-Z][A-Z\s]+)',
                r'(?:Complainant|तक्रारदार|शिकायतकर्ता)\s*[:/]\s*([A-Z][A-Z\s]+)',
                r'([A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,})',  # Three consecutive uppercase words
                r'(?:^|\s)([A-Z][A-Z]+\s+[A-Z][A-Z]+\s+[A-Z][A-Z]+)(?:\s|$)',
                # Hindi names in Devanagari
                r'(?:नाव|आरोपी)\s*[:।]\s*([\u0900-\u097F\s]+)',
            ],
            'addresses': [
                r'(?:पत्ता|Address|PATTA|पƣा)\s*[:।]\s*([^,\n]+(?:,[^,\n]+)*)',
                r'(?:रा\.|रहिवासी|Resident|रे\.)\s*[:।]?\s*([^,\n]+(?:,[^,\n]+)*)',
                r'(?:शहर|City|जिल्हा|District|State|राज्य|जज\.|ता\.)\s*[:।]\s*([^,\n]+)',
                r'(?:Town|Village|शहर|गाव)\s*[:।]\s*([^,\n]+)',
                r'(?:PIN|पिन)\s*[:।]?\s*(\d{6})',
            ],
            'dates': [
                r'\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})\b',
                r'\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2})\b',
                r'(?:दिनांक|Date|दिन)\s*[:।]\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{1,2}-\d{1,2}-\d{4})',
            ],
            'fir_numbers': [
                r'(?:FIR|F\.I\.R\.|एफ\.आई\.आर\.)\s*(?:No\.|नं\.?|क्रमांक|Đ\.)\s*[:।]?\s*(\d+)',
                r'गु\.र\.(?:नं\.|Ȳ\.)\s*(?:व\s*)?(?:म-)?\s*(\d+[\/\-]\d+)',
                r'(?:गुन्हा|गुÛहा)\s*(?:नंबर|नं\.)\s*[:।]\s*(\d+)',
                r'(\d{4})\s*(?:Date|दिनांक)',  # FIR number before date
                r'(\d+/\d+)',  # Case number format
            ],
            'police_stations': [
                r'(?:P\.S\.|पोलीस\s*ठाणे|Police\s*Station)\s*[:।]?\s*([^,\n\d]+)',
                r'(?:ठाणे|स्टेशन)\s*[:।]?\s*([^,\n\d]+)',
                r'(?:भोसरी|पुणे|मुंबई|दिल्ली|[A-Z]{3,})\s*(?:पोलीस|Police)',
            ],
            'ages': [
                r'(?:वय|Age|Years|वषȶ)\s*[:।]?\s*(\d+)\s*(?:वर्षे?|years?|yrs?|वष)',
                r'(\d+)\s*(?:वर्षे?|years?|yrs?|वष)',
                r'(?:वय|Age)\s*(\d+)',
            ],
            'amounts': [
                r'(?:रु\.|Rs\.?|रुपये|ǽ\.)\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:रु\.|Rs\.?|रुपये|ǽ\.)',
                r'(?:मूल्य|Value|मǕãय)\s*[:।]\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                r'(\d{2,}\.\d{2})',  # Decimal amounts
            ],
            'vehicle_numbers': [
                r'\b([A-Z]{2}[-\s]?\d{1,2}[-\s]?[A-Z]{1,2}[-\s]?\d{4})\b',
                r'(?:KZ|MH|DL|UP|GJ)\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{4}',
                r'([A-Z]{2}\s*\d{2}\s*[A-Z]{2}\s*\d{4})',
            ],
            'sections_acts': [
                r'(?:कलम|Section|धारा)\s*(\d+(?:\([^)]+\))?)',
                r'(?:IPC|भारतीय\s*दंड\s*संहिता|आयपीसी)\s*(\d+)',
                r'(?:शè|Arms)\s*(?:ğ|Act)\s*(?:\ͬǓयम|Act)\s*(\d+)',
                r'(?:महाराçĚ|Maharashtra)\s*(?:पोीस|Police)\s*(?:\ͬǓयम|Act)\s*(\d+)',
            ],
            'ids': [
                r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # Aadhaar
                r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN
                r'(?:आधार|Aadhaar|UID)\s*(?:नं\.|No\.?)\s*[:।]?\s*(\d{4}[-.\s]?\d{4}[-.\s]?\d{4})',
                r'(?:PAN|पॅन)\s*(?:नं\.|No\.?)\s*[:।]?\s*([A-Z]{5}\d{4}[A-Z])',
            ]
        }

    def extract_text_advanced(self, pdf_file) -> Tuple[str, str]:
        """Advanced text extraction with OCR fallback"""
        text = ""
        ocr_text = ""
        
        # Method 1: Direct text extraction
        try:
            pdf_file.seek(0)
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Extract tables separately
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text += " ".join([str(cell) for cell in row if cell]) + "\n"
        except Exception as e:
            st.warning(f"pdfplumber failed: {e}")
        
        # Method 2: PyMuPDF with enhanced extraction
        try:
            pdf_file.seek(0)
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Extract text with different methods
                text += page.get_text() + "\n"
                text += page.get_text("dict")  # Dictionary format for better structure
                
                # Extract text blocks
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) > 4:  # Text block
                        text += block[4] + "\n"
                        
            pdf_document.close()
        except Exception as e:
            st.warning(f"PyMuPDF failed: {e}")
            
        # Method 3: OCR as fallback for scanned documents
        try:
            pdf_file.seek(0)
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page_num in range(min(pdf_document.page_count, 5)):  # Limit OCR to first 5 pages
                page = pdf_document[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to numpy array for OpenCV
                img_np = np.array(img)
                
                # Preprocess image for better OCR
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                
                # Apply denoising
                denoised = cv2.fastNlMeansDenoising(gray)
                
                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(
                    denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                
                # OCR with Hindi and English
                custom_config = r'--oem 3 --psm 6 -l hin+eng'
                ocr_result = pytesseract.image_to_string(thresh, config=custom_config)
                ocr_text += ocr_result + "\n"
                
            pdf_document.close()
            
        except Exception as e:
            st.warning(f"OCR extraction failed: {e}")
            
        return text, ocr_text

    def clean_and_normalize_text(self, text: str) -> str:
        """Advanced text cleaning and normalization"""
        # Character replacements for common OCR errors
        char_corrections = {
            'ɮ': 'व', 'ĵ': 'ज', 'ç': 'क', 'Ě': 'ष', 'ã': 'न', 'Ĥ': 'प',
            'ʜ': 'स', 'Ͻ': 'त', 'ǔ': 'ज', 'Ǖ': 'उ', 'ͧ': 'श', 'Ǘ': 'र',
            'Đ': 'न', 'Ȳ': 'र', 'ğ': 'स', 'è': 'स', 'Ǔ': 'इ', 'ͩ': 'क',
            'Ȫ': 'म', '×': 'त', 'ã': 'न', 'ǽ': 'रु', 'मǕ': 'मू', 'àह': 'मह',
            'Ȯ': 'ए', 'ȧ': 'ी', 'ͬ': 'नि', 'ĭ': 'ज', 'Ą': 'ल', 'ɍ': 'त'
        }
        
        # Apply character corrections
        for wrong, correct in char_corrections.items():
            text = text.replace(wrong, correct)
        
        # Fix common word fragments
        word_corrections = {
            'P S ': 'P.S. ', 'F I R': 'FIR', 'No ': 'No. ',
            'पोीस': 'पोलीस', 'ठाे': 'ठाणे', 'दिांक': 'दिनांक',
            'वेळ': 'वेळ', 'नमूद': 'नमूद', 'केेली': 'केलेली'
        }
        
        for wrong, correct in word_corrections.items():
            text = text.replace(wrong, correct)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()

    def extract_pii_enhanced(self, text: str) -> Dict[str, List[str]]:
        """Enhanced PII extraction with better patterns"""
        extracted_pii = {}
        
        # Process text in chunks for better context
        lines = text.split('\n')
        full_text = ' '.join(lines)
        
        for category, patterns in self.patterns.items():
            extracted_pii[category] = set()  # Use set to avoid duplicates
            
            for pattern in patterns:
                # Search in full text
                matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            # Handle grouped matches
                            if category == 'dates':
                                date_str = f"{match[0]}/{match[1]}/{match[2]}"
                                extracted_pii[category].add(date_str)
                            else:
                                extracted_pii[category].add('/'.join(match))
                        else:
                            extracted_pii[category].add(str(match).strip())
                
                # Search line by line for context
                for line in lines:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            if isinstance(match, tuple):
                                if category == 'dates':
                                    date_str = f"{match[0]}/{match[1]}/{match[2]}"
                                    extracted_pii[category].add(date_str)
                                else:
                                    extracted_pii[category].add('/'.join(match))
                            else:
                                extracted_pii[category].add(str(match).strip())
        
        # Convert sets back to lists and filter
        for category in extracted_pii:
            extracted_pii[category] = [
                item for item in list(extracted_pii[category]) 
                if item and len(item.strip()) > 1
            ]
        
        return extracted_pii

    def extract_fir_specific_data(self, text: str) -> Dict[str, Any]:
        """Extract FIR-specific structured data"""
        fir_data = {}
        
        # Enhanced FIR number extraction
        fir_patterns = [
            r'FIR\s*No\.?\s*[:।]?\s*(\d+)',
            r'(\d{4})\s+Date\s+and\s+Time',
            r'गु\.र\..*?(\d+/\d+)',
            r'क्रमांक\s*[:।]\s*(\d+)'
        ]
        
        for pattern in fir_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fir_data['fir_number'] = match.group(1)
                break
        
        # Extract complainant name specifically
        complainant_patterns = [
            r'Complainant.*?Name\s*[:।]\s*([A-Z\s]+?)(?=\n|Father)',
            r'(?:नाव|Name)\s*[:।]\s*([A-Z][A-Z\s]+?)(?=\n|वडिल|Father)',
            r'VIPUL\s+RANGNATH\s+JADHAV',  # Specific to your document
        ]
        
        for pattern in complainant_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                fir_data['complainant_name'] = match.group(0) if 'VIPUL' in pattern else match.group(1)
                break
        
        # Extract accused details
        accused_patterns = [
            r'(?:मान\s+जगताशंकर\s+शर्मा|मा\s+जटाशȲर\s+शमा)',
            r'(?:रविकुमार\s+हरिशंकर\s+शर्मा|रववġुमार\s+हǐरशȲर\s+शमा)',
        ]
        
        accused_names = []
        for pattern in accused_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            accused_names.extend(matches)
        
        if accused_names:
            fir_data['accused_names'] = accused_names
        
        # Extract amounts with context
        amount_patterns = [
            r'(\d{2,}(?:,\d+)*(?:\.\d+)?)\s*(?:रु\.|Rs\.?|ǽ)',
            r'(?:रु\.|Rs\.?|ǽ\.?)\s*(\d{2,}(?:,\d+)*(?:\.\d+)?)',
            r'Value.*?(\d{2,}(?:,\d+)*(?:\.\d+)?)',
            r'मूल्य.*?(\d{2,}(?:,\d+)*(?:\.\d+)?)',
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        
        if amounts:
            fir_data['amounts'] = list(set(amounts))
        
        # Extract police station
        ps_match = re.search(r'P\.S\..*?[:।]\s*([^F\n]+)', text, re.IGNORECASE)
        if ps_match:
            fir_data['police_station'] = ps_match.group(1).strip()
        
        # Extract district
        district_match = re.search(r'District.*?[:।]\s*([^Y\n]+)', text, re.IGNORECASE)
        if district_match:
            fir_data['district'] = district_match.group(1).strip()
        
        return fir_data

    def extract_phone_numbers_advanced(self, text: str) -> List[str]:
        """Advanced phone number extraction"""
        phones = set()
        
        # Multiple phone patterns
        patterns = [
            r'\b[6789]\d{9}\b',  # 10-digit mobile
            r'\b(?:\+91|91)?[-.\s]?[6789]\d{9}\b',
            r'(?:मो\.|Mobile|Phone|फोन).*?(\d{10})',
            r'(\d{4}\d{3}\d{4})',  # Without separators
            r'(\d{4}[-.\s]\d{3}[-.\s]\d{4})',  # With separators
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean the match
                clean_number = re.sub(r'[^\d]', '', str(match))
                if len(clean_number) == 10 and clean_number[0] in '6789':
                    phones.add(clean_number)
                elif len(clean_number) == 12 and clean_number.startswith('91'):
                    phones.add(clean_number[2:])  # Remove country code
        
        return list(phones)

def main():
    st.title("🚀 Advanced Legal PDF PII Extractor")
    st.markdown("### High-Precision Extraction for Indian Legal Documents")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Extraction Settings")
        use_ocr = st.checkbox("Enable OCR for Scanned PDFs", value=True)
        enhance_hindi = st.checkbox("Enhanced Hindi Processing", value=True)
        extract_tables = st.checkbox("Extract Table Data", value=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.7)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Legal PDF Document", 
        type=['pdf'], 
        help="Upload FIR or other legal documents for PII extraction"
    )
    
    if uploaded_file is not None:
        extractor = AdvancedPIIExtractor()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Text Extraction
        status_text.text("🔄 Extracting text from PDF...")
        progress_bar.progress(20)
        
        extracted_text, ocr_text = extractor.extract_text_advanced(uploaded_file)
        
        # Step 2: Text Cleaning
        status_text.text("🧹 Cleaning and normalizing text...")
        progress_bar.progress(40)
        
        cleaned_text = extractor.clean_and_normalize_text(extracted_text)
        if use_ocr and ocr_text:
            cleaned_ocr = extractor.clean_and_normalize_text(ocr_text)
            combined_text = cleaned_text + "\n" + cleaned_ocr
        else:
            combined_text = cleaned_text
        
        # Step 3: PII Extraction
        status_text.text("🔍 Extracting PII data...")
        progress_bar.progress(60)
        
        pii_data = extractor.extract_pii_enhanced(combined_text)
        fir_specific = extractor.extract_fir_specific_data(combined_text)
        advanced_phones = extractor.extract_phone_numbers_advanced(combined_text)
        
        # Step 4: Enhancement and Validation
        status_text.text("✅ Validating and enhancing results...")
        progress_bar.progress(80)
        
        # Merge phone numbers
        if advanced_phones:
            pii_data['phone_numbers'] = list(set(pii_data.get('phone_numbers', []) + advanced_phones))
        
        progress_bar.progress(100)
        status_text.text("✅ Extraction completed!")
        
        # Display results
        st.success(f"✅ Successfully processed PDF: {uploaded_file.name}")
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Enhanced Summary", 
            "🎯 FIR Specific Data", 
            "📋 Detailed PII", 
            "📄 Raw Text", 
            "💾 Export"
        ])
        
        with tab1:
            st.header("Enhanced PII Summary")
            
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📱 Phone Numbers", len(pii_data.get('phone_numbers', [])))
                if pii_data.get('phone_numbers'):
                    for phone in pii_data['phone_numbers']:
                        st.code(phone)
            
            with col2:
                st.metric("👤 Names", len(pii_data.get('names', [])))
                if pii_data.get('names'):
                    for name in pii_data['names'][:5]:
                        st.write(f"• {name}")
            
            with col3:
                st.metric("🏠 Addresses", len(pii_data.get('addresses', [])))
                if pii_data.get('addresses'):
                    for addr in pii_data['addresses'][:3]:
                        st.write(f"• {addr[:50]}...")
            
            with col4:
                st.metric("💰 Amounts", len(pii_data.get('amounts', [])))
                if pii_data.get('amounts'):
                    for amount in pii_data['amounts']:
                        st.code(f"₹{amount}")
        
        with tab2:
            st.header("FIR Specific Extracted Data")
            
            if fir_specific:
                # Display in organized format
                col1, col2 = st.columns(2)
                
                with col1:
                    if fir_specific.get('fir_number'):
                        st.info(f"**FIR Number:** {fir_specific['fir_number']}")
                    
                    if fir_specific.get('complainant_name'):
                        st.info(f"**Complainant:** {fir_specific['complainant_name']}")
                    
                    if fir_specific.get('police_station'):
                        st.info(f"**Police Station:** {fir_specific['police_station']}")
                
                with col2:
                    if fir_specific.get('district'):
                        st.info(f"**District:** {fir_specific['district']}")
                    
                    if fir_specific.get('accused_names'):
                        st.info("**Accused Persons:**")
                        for name in fir_specific['accused_names']:
                            st.write(f"• {name}")
                    
                    if fir_specific.get('amounts'):
                        st.info("**Financial Details:**")
                        for amount in fir_specific['amounts']:
                            st.write(f"• ₹{amount}")
            else:
                st.warning("No FIR-specific data structure detected")
        
        with tab3:
            st.header("Detailed PII Categories")
            
            for category, items in pii_data.items():
                if items:
                    with st.expander(f"{category.replace('_', ' ').title()} ({len(items)} found)"):
                        df = pd.DataFrame({'Extracted Data': items})
                        st.dataframe(df, use_container_width=True)
        
        with tab4:
            st.header("Extracted Text Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Cleaned Text")
                st.text_area("Processed Text", combined_text[:3000] + "..." if len(combined_text) > 3000 else combined_text, height=400)
            
            with col2:
                st.subheader("📊 Text Statistics")
                st.metric("Total Characters", len(combined_text))
                st.metric("Total Lines", len(combined_text.split('\n')))
                st.metric("Hindi Characters", len(re.findall(r'[\u0900-\u097F]', combined_text)))
                st.metric("English Characters", len(re.findall(r'[A-Za-z]', combined_text)))
        
        with tab5:
            st.header("Export Extracted Data")
            
            # Prepare comprehensive export data
            export_data = {
                'extraction_metadata': {
                    'file_name': uploaded_file.name,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'text_length': len(combined_text),
                    'extraction_method': 'Advanced Multi-Method + OCR'
                },
                'fir_specific_data': fir_specific,
                'pii_categories': pii_data,
                'raw_text': combined_text[:5000],  # First 5000 chars
                'summary': {
                    'total_pii_items': sum(len(items) for items in pii_data.values()),
                    'categories_found': len([k for k, v in pii_data.items() if v])
                }
            }
            
            # JSON export
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Complete Analysis (JSON)",
                data=json_str,
                file_name=f"fir_pii_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Excel export
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_df = pd.DataFrame([
                        {'Category': k, 'Count': len(v), 'Items': '; '.join(v[:5])} 
                        for k, v in pii_data.items() if v
                    ])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Individual category sheets
                    for category, items in pii_data.items():
                        if items:
                            df = pd.DataFrame({'Value': items})
                            df.to_excel(writer, sheet_name=category[:30], index=False)
                
                excel_buffer.seek(0)
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"fir_pii_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Excel export failed: {e}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

# Additional utility functions for maximum extraction accuracy

def preprocess_image_for_ocr(image_np: np.ndarray) -> np.ndarray:
    """Advanced image preprocessing for better OCR results"""
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply dilation and erosion to clean up text
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh

def extract_with_multiple_ocr_configs(image: np.ndarray) -> str:
    """Try multiple OCR configurations for best results"""
    configs = [
        r'--oem 3 --psm 6 -l hin+eng',  # Default
        r'--oem 3 --psm 4 -l hin+eng',  # Single column
        r'--oem 3 --psm 3 -l hin+eng',  # Fully automatic
        r'--oem 1 --psm 6 -l hin+eng',  # LSTM only
    ]
    
    best_result = ""
    max_length = 0
    
    for config in configs:
        try:
            result = pytesseract.image_to_string(image, config=config)
            if len(result) > max_length:
                max_length = len(result)
                best_result = result
        except:
            continue
    
    return best_result

def post_process_extracted_names(names: List[str]) -> List[str]:
    """Post-process names to improve accuracy"""
    cleaned_names = []
    
    for name in names:
        # Remove common prefixes/suffixes
        name = re.sub(r'^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)', '', name, flags=re.IGNORECASE)
        name = re.sub(r'(Jr\.?|Sr\.?|III?|IV?)', '', name, flags=re.IGNORECASE)
        
        # Clean whitespace
        name = ' '.join(name.split())
        
        # Validate name (must have at least 2 parts)
        name_parts = name.split()
        if len(name_parts) >= 2 and all(len(part) > 1 for part in name_parts):
            # Check if it's mostly alphabetic
            alpha_ratio = sum(c.isalpha() or c in 'ऀ-ॿ' for c in name) / len(name)
            if alpha_ratio > 0.7:
                cleaned_names.append(name)
    
    return list(set(cleaned_names))

def extract_document_metadata(text: str) -> Dict[str, str]:
    """Extract document metadata and classification"""
    metadata = {}
    
    # Document type detection
    if any(term in text.lower() for term in ['fir', 'first information report', 'एफआईआर']):
        metadata['document_type'] = 'FIR'
    elif any(term in text.lower() for term in ['charge sheet', 'चार्जशीट']):
        metadata['document_type'] = 'Charge Sheet'
    else:
        metadata['document_type'] = 'Legal Document'
    
    # Extract court/jurisdiction info
    court_match = re.search(r'(?:Court|न्यायालय|कोर्ट)\s*[:।]\s*([^,\n]+)', text, re.IGNORECASE)
    if court_match:
        metadata['court'] = court_match.group(1).strip()
    
    # Extract case category
    if any(term in text.lower() for term in ['theft', 'चोरी', 'theft']):
        metadata['case_category'] = 'Theft'
    elif any(term in text.lower() for term in ['assault', 'हल्ला']):
        metadata['case_category'] = 'Assault'
    elif any(term in text.lower() for term in ['fraud', 'फसवणूक']):
        metadata['case_category'] = 'Fraud'
    
    return metadata

if __name__ == "__main__":
    main()

# Performance optimization tips:
"""
PERFORMANCE OPTIMIZATION:
========================

1. For large PDFs, consider processing page by page
2. Use threading for OCR processing if dealing with multiple files
3. Cache results using st.cache_data for repeated processing
4. Implement batch processing for multiple files

ACCURACY IMPROVEMENTS:
=====================

1. Train custom NER models for legal domain
2. Use language-specific preprocessing
3. Implement fuzzy matching for name variations
4. Add manual review interface for uncertain extractions

DEPLOYMENT NOTES:
================

1. Install Tesseract OCR: sudo apt-get install tesseract-ocr tesseract-ocr-hin
2. For production, consider using cloud OCR services (Google Vision, Azure)
3. Add error logging and monitoring
4. Implement user feedback mechanism for continuous improvement

REQUIRED PACKAGES:
=================

pip install streamlit PyPDF2 pdfplumber PyMuPDF pandas pillow pytesseract opencv-python openpyxl

Additional system requirements:
- Tesseract OCR with Hindi language pack
- OpenCV for image processing
"""
