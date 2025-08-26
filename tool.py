import streamlit as st
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Set, Any
import json

class SmartPIIExtractor:
    def __init__(self):
        # Define field mappings with multiple language variants
        self.field_patterns = {
            'name': [
                r'Name\s*\([^)]*\):\s*([A-Z][A-Z\s]+)',
                r'नाव\):\s*([A-Z][A-Z\s]+)',
                r'Father.*Name[^:]*:\s*([A-Z][A-Z\s]+)',
                r'वडिलांचे.*नाव[^:]*:\s*([A-Z][A-Z\s]+)',
            ],
            'fir_number': [
                r'FIR\s+No[^:]*:\s*(\d+)',
                r'खब[^:]*:\s*(\d+)',
            ],
            'date': [
                r'Date[^:]*:\s*(\d{1,2}/\d{1,2}/\d{4})',
                r'दिनांक[^:]*:\s*(\d{1,2}/\d{1,2}/\d{4})',
            ],
            'time': [
                r'Time[^:]*:\s*(\d{1,2}:\d{2})',
                r'वेळ[^:]*:\s*(\d{1,2}:\d{2})',
            ],
            'police_station': [
                r'P\.S\.[^:]*:\s*([^\n\\]+)',
                r'पोलीस\s+ठाणे[^:]*:\s*([^\n\\]+)',
            ],
            'district': [
                r'District[^:]*:\s*([^\n\\]+)',
                r'जिल्हा[^:]*:\s*([^\n\\]+)',
            ],
            'address': [
                r'Address[^:]*:\s*([^\n]+(?:\n[^a-zA-Z\n]+)*)',
                r'पत्ता[^:]*:\s*([^\n]+(?:\n[^a-zA-Z\n]+)*)',
            ],
            'year': [
                r'Year[^:]*:\s*(\d{4})',
                r'वर्ष[^:]*:\s*(\d{4})',
            ],
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize the input text"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Clean up common encoding issues
        text = re.sub(r'[\\]+n', '\n', text)
        # Normalize colons
        text = re.sub(r':\s*', ': ', text)
        return text.strip()
    
    def extract_field_values(self, text: str) -> Dict[str, List[str]]:
        """Extract values using field-based approach"""
        results = {}
        
        # Preprocess text
        clean_text = self.preprocess_text(text)
        
        for field_type, patterns in self.field_patterns.items():
            values = set()
            
            for pattern in patterns:
                matches = re.findall(pattern, clean_text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    # Clean the extracted value
                    cleaned_match = self.clean_extracted_value(match, field_type)
                    if cleaned_match:
                        values.add(cleaned_match)
            
            if values:
                results[field_type] = list(values)
        
        return results
    
    def clean_extracted_value(self, value: str, field_type: str) -> str:
        """Clean and validate extracted values"""
        if not value:
            return ""
        
        # Basic cleaning
        value = value.strip()
        value = re.sub(r'\\n.*$', '', value)  # Remove everything after \n
        value = re.sub(r'\s+', ' ', value)    # Normalize spaces
        
        # Type-specific cleaning
        if field_type == 'name':
            # Must be reasonable name format
            if len(value) < 4 or len(value) > 50:
                return ""
            if not re.match(r'^[A-Z][A-Z\s]*$', value):
                return ""
            # Remove common false positives
            false_positives = ['NAME OF', 'TIME OF', 'DATE AND', 'PLACE OF', 'TYPE OF']
            if value.strip() in false_positives:
                return ""
        
        elif field_type in ['police_station', 'district']:
            # Remove leading symbols and clean
            value = re.sub(r'^[^\w\u0900-\u097F]*', '', value)
            if len(value) < 2:
                return ""
        
        elif field_type == 'address':
            # Must be reasonable address length
            if len(value) < 10 or len(value) > 200:
                return ""
        
        elif field_type == 'date':
            # Validate date format
            if not re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', value):
                return ""
        
        elif field_type == 'time':
            # Validate time format
            if not re.match(r'^\d{1,2}:\d{2}$', value):
                return ""
        
        return value.strip()
    
    def extract_additional_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract additional PII using pattern matching"""
        additional_pii = {}
        
        # Phone numbers
        phone_patterns = [
            r'\+91[\s-]?([6-9]\d{9})',
            r'\b([6-9]\d{9})\b',
            r'Phone[^:]*:\s*(\+?91?[\s-]?[6-9]\d{9})',
        ]
        
        phones = set()
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) >= 10:
                    phones.add(match)
        
        if phones:
            additional_pii['phone_numbers'] = list(phones)
        
        # Entry numbers
        entry_matches = re.findall(r'Entry\s+No[^:]*:\s*(\d+)', text, re.IGNORECASE)
        if entry_matches:
            additional_pii['entry_numbers'] = entry_matches
        
        # Beat numbers  
        beat_matches = re.findall(r'Beat\s+No[^:]*:\s*(\d+)', text, re.IGNORECASE)
        if beat_matches:
            additional_pii['beat_numbers'] = beat_matches
        
        # Section numbers
        section_matches = re.findall(r'Section(?:s)?\s*[^:]*:\s*(\d+(?:\(\d+\))?)', text, re.IGNORECASE)
        if section_matches:
            additional_pii['legal_sections'] = section_matches
        
        return additional_pii
    
    def extract_contextual_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using contextual understanding"""
        entities = {}
        
        # Look for standalone proper nouns that might be names
        lines = text.split('\n')
        potential_names = set()
        
        for line in lines:
            # Skip lines that are clearly field labels
            if ':' in line and any(keyword in line.lower() for keyword in ['name', 'नाव', 'father', 'वडिल']):
                # Extract value after colon
                parts = line.split(':', 1)
                if len(parts) > 1:
                    value = parts[1].strip()
                    if re.match(r'^[A-Z][A-Z\s]{3,40}$', value):
                        potential_names.add(value)
        
        if potential_names:
            entities['extracted_names'] = list(potential_names)
        
        # Look for Indian city names
        indian_cities = ['पुणे', 'मुंबई', 'दिल्ली', 'भोसरी', 'कोल्हापूर', 'नागपूर', 'औरंगाबाद']
        found_cities = []
        for city in indian_cities:
            if city in text:
                found_cities.append(city)
        
        if found_cities:
            entities['cities'] = found_cities
        
        return entities
    
    def extract_all_pii(self, text: str) -> Dict[str, List[str]]:
        """Main extraction method combining all approaches"""
        all_pii = {}
        
        # Method 1: Field-based extraction
        field_pii = self.extract_field_values(text)
        all_pii.update(field_pii)
        
        # Method 2: Additional pattern matching
        additional_pii = self.extract_additional_patterns(text)
        all_pii.update(additional_pii)
        
        # Method 3: Contextual entity extraction
        contextual_pii = self.extract_contextual_entities(text)
        all_pii.update(contextual_pii)
        
        return all_pii

def main():
    st.set_page_config(
        page_title="Smart PII Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Smart PII Extractor Tool")
    st.markdown("Advanced Personal Identifiable Information extraction using NLP techniques")
    
    # Initialize extractor
    extractor = SmartPIIExtractor()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Extraction Methods")
        st.markdown("""
        **🎯 Field-Based Extraction**
        - Targets specific form fields
        - Multi-language support
        - Context-aware cleaning
        
        **🔍 Pattern Recognition**
        - Phone numbers
        - Reference numbers
        - Legal sections
        
        **🧠 Contextual Analysis**
        - Proper noun detection
        - Geographic entities
        - Relationship mapping
        """)
        
        st.header("🔧 Detected Categories")
        st.markdown("""
        - **Names**: Person names
        - **Dates & Times**: Timestamps
        - **FIR Numbers**: Case references
        - **Police Stations**: Jurisdiction
        - **Districts**: Geographic regions
        - **Addresses**: Physical locations
        - **Phone Numbers**: Contact info
        - **Legal Sections**: Act references
        """)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Sample FIR text
        sample_text = """P.S. (पोलीस ठाणे): भोसरी
FIR No. (FIR खब Đ.): 0523
Date and Time of FIR (FIR ख. दिनांक आणण वेळ): 19/11/2017 21:33 वाजता
District (जिल्हा): पुणे शहर
Year (वर्ष): 2017

Complainant / Informant (तक्रारदार / माहिती देणारा):
(a) Name (नाव): VIPUL RANGNATH JADHAV
(b) Father's/Husband's Name (वडिलांचे/पतीचे नाव): RANGNATH JADHAV

Place of Occurrence (घटनास्थळ):
Address (पत्ता): अशोक हॉटेल्या पाठीमागे, मोरया मंदिरात, राशिवाटा, वासारवाडी, पुणे

Information received at P.S. (पोलीस ठाण्यावर माहिती मिळाल्याचा):
Date (दिनांक): 19/11/2017
Time (वेळ): 21:09 तास

Entry No. (नोंद Đ.): 029
Beat No. (बीट Đ.): 1

Phone: +91 9876543210
Case No: 123/2017"""
        
        # Input methods
        input_method = st.radio("Choose input method:", ["Paste Text", "Use Sample", "Upload File"])
        
        if input_method == "Use Sample":
            text_input = sample_text
            st.text_area("Sample FIR text:", text_input, height=400, disabled=True)
        elif input_method == "Paste Text":
            text_input = st.text_area("Paste your text here:", height=400)
        else:
            uploaded_file = st.file_uploader("Upload text file:", type=['txt', 'log'])
            if uploaded_file:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", text_input[:500] + "...", height=200)
            else:
                text_input = ""
    
    with col2:
        st.header("🎯 Extracted PII")
        
        if st.button("🚀 Extract PII", type="primary") and text_input:
            with st.spinner("Processing with smart extraction..."):
                # Extract PII
                pii_results = extractor.extract_all_pii(text_input)
                
                if pii_results:
                    st.success(f"✅ Found PII in {len(pii_results)} categories")
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📊 Results", "🔍 Details", "💾 Export"])
                    
                    with tab1:
                        st.subheader("📊 Extraction Results")
                        
                        for category, values in pii_results.items():
                            with st.expander(f"**{category.replace('_', ' ').title()}** ({len(values)} items)", expanded=True):
                                for i, value in enumerate(values, 1):
                                    st.write(f"**{i}.** `{value}`")
                    
                    with tab2:
                        st.subheader("🔍 Detailed Analysis")
                        
                        # Create structured view
                        for category, values in pii_results.items():
                            st.write(f"**{category.replace('_', ' ').title()}:**")
                            df = pd.DataFrame({
                                'Value': values,
                                'Length': [len(v) for v in values],
                                'Type': [category] * len(values)
                            })
                            st.dataframe(df, use_container_width=True)
                            st.write("---")
                    
                    with tab3:
                        st.subheader("💾 Export Options")
                        
                        # Prepare export data
                        export_rows = []
                        for category, values in pii_results.items():
                            for value in values:
                                export_rows.append({
                                    'Category': category.replace('_', ' ').title(),
                                    'Value': value,
                                    'Length': len(value),
                                    'Extracted_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                        
                        if export_rows:
                            df_export = pd.DataFrame(export_rows)
                            
                            col_e1, col_e2 = st.columns(2)
                            
                            with col_e1:
                                # CSV export
                                csv = df_export.to_csv(index=False)
                                st.download_button(
                                    "📥 Download CSV",
                                    csv,
                                    f"smart_pii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv"
                                )
                            
                            with col_e2:
                                # JSON export
                                json_data = json.dumps(pii_results, indent=2, ensure_ascii=False)
                                st.download_button(
                                    "📥 Download JSON",
                                    json_data,
                                    f"smart_pii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    "application/json"
                                )
                            
                            st.dataframe(df_export, use_container_width=True)
                    
                    # Summary metrics
                    total_items = sum(len(values) for values in pii_results.values())
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("Total PII Items", total_items)
                    with col_m2:
                        st.metric("Categories", len(pii_results))
                    with col_m3:
                        st.metric("Text Length", f"{len(text_input):,} chars")
                
                else:
                    st.warning("⚠️ No PII detected. Please check your input text.")
        
        elif not text_input:
            st.info("👆 Select an input method and provide text to extract PII")
    
    # Footer
    st.markdown("---")
    st.markdown("**Smart PII Extractor** - Uses advanced NLP techniques for accurate PII detection from legal documents")

if __name__ == "__main__":
    main()
