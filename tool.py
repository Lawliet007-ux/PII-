import streamlit as st
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Set

class PIIExtractor:
    def __init__(self):
        # Patterns for different types of PII
        self.patterns = {
            'names': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names (3 parts)
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names (2 parts)
                r'Name \(नाव\):\s*([A-Z\s]+)',  # Specific pattern from FIR
                r'नाव:\s*([A-Z\s]+)',  # Hindi pattern
            ],
            'dates': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # DD/MM/YYYY or DD-MM-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
                r'Date \(दिनांक\):\s*(\d{1,2}/\d{1,2}/\d{4})',  # Specific FIR date pattern
                r'दिनांक:\s*(\d{1,2}/\d{1,2}/\d{4})',  # Hindi date pattern
            ],
            'addresses': [
                r'Address \(पत्ता\):\s*([^,\n]+(?:,[^,\n]+)*)',  # Address pattern from FIR
                r'पत्ता:\s*([^,\n]+(?:,[^,\n]+)*)',  # Hindi address pattern
                r'\b(?:पुणे|मुंबई|दिल्ली|कोल्हापूर|नागपूर|औरंगाबाद)\b',  # Indian cities
                r'\b\d{6}\b',  # PIN codes
            ],
            'phone_numbers': [
                r'\b(?:\+91|91)?\s*[6-9]\d{9}\b',  # Indian mobile numbers
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # General phone pattern
            ],
            'fir_numbers': [
                r'FIR No\.\s*\([^)]+\):\s*(\d+)',  # FIR number pattern
                r'FIR\s*(?:No|Number)\.?\s*:?\s*(\d+)',  # Alternative FIR patterns
            ],
            'case_numbers': [
                r'Case No\.\s*:?\s*(\d+/\d{4})',  # Case number pattern
                r'केस नं\.:\s*(\d+/\d{4})',  # Hindi case number
            ],
            'police_stations': [
                r'P\.S\.\s*\([^)]+\):\s*([^,\n]+)',  # Police station from FIR
                r'Police Station:\s*([^,\n]+)',  # Alternative police station
                r'पोलीस ठाणे:\s*([^,\n]+)',  # Hindi police station
            ],
            'districts': [
                r'District \([^)]+\):\s*([^,\n]+)',  # District from FIR
                r'जिल्हा:\s*([^,\n]+)',  # Hindi district
            ],
            'times': [
                r'\b\d{1,2}:\d{2}\s*(?:AM|PM|तास|वाजता)?\b',  # Time patterns
                r'Time \([^)]+\):\s*(\d{1,2}:\d{2}[^,\n]*)',  # Time from FIR
            ],
            'identification_numbers': [
                r'\b[A-Z]{2}\d{2}[A-Z]{2}\d{4}\b',  # Vehicle number pattern
                r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b',  # Card numbers (masked)
                r'\b[A-Z]{5}\d{4}[A-Z]\b',  # PAN card pattern
            ]
        }
    
    def extract_pii(self, text: str) -> Dict[str, Set[str]]:
        """Extract PII from text using regex patterns"""
        pii_data = {}
        
        for category, patterns in self.patterns.items():
            found_items = set()
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Handle tuples from group captures
                    for match in matches:
                        if isinstance(match, tuple):
                            for item in match:
                                if item.strip():
                                    found_items.add(item.strip())
                        else:
                            found_items.add(match.strip())
            
            if found_items:
                pii_data[category] = found_items
        
        return pii_data
    
    def clean_and_validate_pii(self, pii_data: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Clean and validate extracted PII"""
        cleaned_pii = {}
        
        for category, items in pii_data.items():
            cleaned_items = []
            
            for item in items:
                # Basic cleaning
                item = item.strip()
                
                # Skip very short items (likely false positives)
                if len(item) < 2:
                    continue
                
                # Category-specific validation
                if category == 'names':
                    # Skip items that are likely not names
                    if not re.match(r'^[A-Za-z\s]+$', item) or len(item) > 50:
                        continue
                elif category == 'phone_numbers':
                    # Validate phone number format
                    if not re.match(r'^\+?[\d\s\-\.]{10,15}$', item):
                        continue
                elif category == 'dates':
                    # Basic date validation
                    if not re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$|^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', item):
                        continue
                
                cleaned_items.append(item)
            
            if cleaned_items:
                cleaned_pii[category] = cleaned_items
        
        return cleaned_pii

def main():
    st.set_page_config(
        page_title="PII Extractor Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Personal Identifiable Information (PII) Extractor")
    st.markdown("Extract PII from Elasticsearch text data")
    
    # Initialize PII extractor
    extractor = PIIExtractor()
    
    # Sidebar with information
    with st.sidebar:
        st.header("📋 PII Categories Detected")
        st.markdown("""
        - **Names**: Person names
        - **Dates**: Various date formats
        - **Addresses**: Physical addresses
        - **Phone Numbers**: Mobile/landline numbers
        - **FIR Numbers**: First Information Report numbers
        - **Case Numbers**: Legal case numbers
        - **Police Stations**: Police station names
        - **Districts**: District information
        - **Times**: Time stamps
        - **ID Numbers**: Various identification numbers
        """)
        
        st.header("🔧 Features")
        st.markdown("""
        - Supports English and Hindi text
        - Handles FIR document formats
        - Validates extracted data
        - Exportable results
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["Paste Text", "Upload File"]
        )
        
        text_input = ""
        
        if input_method == "Paste Text":
            text_input = st.text_area(
                "Paste your Elasticsearch text here:",
                height=400,
                placeholder="Paste your text here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'log']
            )
            if uploaded_file is not None:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content preview:", text_input[:500] + "...", height=200)
        
        # Sample text button
        if st.button("🔄 Load Sample FIR Text"):
            sample_text = """P.S. (पोलीस ठाणे): भोसरी
FIR No. (Ĥम खब Đ.): 0523
Date and Time of FIR (Ĥ. ख. दिनांक आणण वेळ): 19/11/2017 21:33 वाजता
District (ǔजã¡ा): पुणे शहर
Year (वष[): 2017

Complainant / Informant (Đािा / माद¡ि िेणाा):
Name (नाव): VIPUL RANGNATH JADHAV
Father's/Husband's Name (वडिलांचे/पिचे नाव): RANGNATH JADHAV

Place of Occurrence (घटनाèळ):
Address (पƣा): \\शोा हटेÍया पाठȤमा, मोया मȰाात, ाशिाटा, ासारवाडि, पुणे

Phone: +91 9876543210
Case No: 123/2017"""
            st.text_area("Sample text loaded:", sample_text, height=200, key="sample")
            text_input = sample_text
    
    with col2:
        st.header("🎯 Extracted PII")
        
        if st.button("🔍 Extract PII", type="primary") and text_input:
            with st.spinner("Extracting PII..."):
                # Extract PII
                raw_pii = extractor.extract_pii(text_input)
                cleaned_pii = extractor.clean_and_validate_pii(raw_pii)
                
                if cleaned_pii:
                    # Display results
                    st.success(f"Found PII in {len(cleaned_pii)} categories")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📋 Detailed View", "💾 Export"])
                    
                    with tab1:
                        # Summary statistics
                        st.subheader("📊 PII Summary")
                        summary_data = []
                        total_items = 0
                        
                        for category, items in cleaned_pii.items():
                            count = len(items)
                            total_items += count
                            summary_data.append({
                                "Category": category.replace('_', ' ').title(),
                                "Count": count,
                                "Examples": ', '.join(list(items)[:3]) + ('...' if len(items) > 3 else '')
                            })
                        
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True)
                        
                        # Metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Total PII Items", total_items)
                        with col_m2:
                            st.metric("Categories Found", len(cleaned_pii))
                        with col_m3:
                            st.metric("Text Length", f"{len(text_input)} chars")
                    
                    with tab2:
                        # Detailed view
                        st.subheader("📋 Detailed PII Information")
                        
                        for category, items in cleaned_pii.items():
                            with st.expander(f"{category.replace('_', ' ').title()} ({len(items)} items)"):
                                for i, item in enumerate(items, 1):
                                    st.write(f"{i}. {item}")
                    
                    with tab3:
                        # Export options
                        st.subheader("💾 Export Results")
                        
                        # Prepare data for export
                        export_data = []
                        for category, items in cleaned_pii.items():
                            for item in items:
                                export_data.append({
                                    "Category": category.replace('_', ' ').title(),
                                    "Value": item,
                                    "Length": len(item),
                                    "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                        
                        if export_data:
                            df_export = pd.DataFrame(export_data)
                            
                            # CSV download
                            csv = df_export.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name=f"pii_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
                            # JSON download
                            import json
                            json_data = json.dumps(cleaned_pii, indent=2, default=list)
                            st.download_button(
                                label="📥 Download as JSON",
                                data=json_data,
                                file_name=f"pii_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            
                            st.dataframe(df_export, use_container_width=True)
                
                else:
                    st.warning("No PII found in the provided text. Try adjusting the input or check the text format.")
        
        elif not text_input:
            st.info("👆 Please enter text in the input area to extract PII")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note**: This tool is designed for security analysis and compliance purposes. Always handle PII responsibly and in accordance with applicable privacy laws.")

if __name__ == "__main__":
    main()
