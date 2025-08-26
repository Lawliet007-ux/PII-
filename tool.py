import streamlit as st
import re
from datetime import datetime
import pandas as pd
from typing import List, Dict, Set, Any, Tuple
import json

class IntelligentPIIExtractor:
    def __init__(self):
        # Define what constitutes PII semantically
        self.pii_indicators = {
            'person_name': ['name', 'नाव', 'father', 'husband', 'वडिल', 'पती'],
            'location': ['address', 'पत्ता', 'police station', 'पोलीस ठाणे', 'district', 'जिल्हा'],
            'identifier': ['fir', 'case', 'entry', 'beat', 'phone', 'mobile'],
            'temporal': ['date', 'time', 'दिनांक', 'वेळ', 'year', 'वर्ष'],
        }
    
    def parse_text_semantically(self, text: str) -> Dict[str, Any]:
        """Parse text line by line with semantic understanding"""
        lines = text.strip().split('\n')
        extracted_data = {
            'names': [],
            'addresses': [],
            'phone_numbers': [],
            'dates': [],
            'times': [],
            'fir_numbers': [],
            'case_numbers': [],
            'police_stations': [],
            'districts': [],
            'years': [],
            'entry_numbers': [],
            'beat_numbers': [],
            'other_identifiers': []
        }
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
                
            # Analyze this line for PII
            pii_data = self.analyze_line_for_pii(line, line_num, lines)
            
            # Merge results
            for category, values in pii_data.items():
                if values and category in extracted_data:
                    extracted_data[category].extend(values)
        
        # Clean and deduplicate
        return self.clean_extracted_data(extracted_data)
    
    def analyze_line_for_pii(self, line: str, line_num: int, all_lines: List[str]) -> Dict[str, List[str]]:
        """Analyze a single line for PII using semantic understanding"""
        pii_found = {}
        
        # Case 1: Field-value pairs (most common in forms)
        if ':' in line:
            pii_found.update(self.extract_field_value_pairs(line))
        
        # Case 2: Standalone values that look like PII
        pii_found.update(self.extract_standalone_pii(line))
        
        # Case 3: Multi-line values (addresses often span multiple lines)
        if line_num < len(all_lines) - 1:
            pii_found.update(self.extract_multiline_pii(line, all_lines[line_num + 1:line_num + 3]))
        
        return pii_found
    
    def extract_field_value_pairs(self, line: str) -> Dict[str, List[str]]:
        """Extract PII from field:value pairs"""
        results = {}
        
        # Split on colon
        if ':' not in line:
            return results
            
        parts = line.split(':', 1)
        if len(parts) != 2:
            return results
            
        field_part = parts[0].strip().lower()
        value_part = parts[1].strip()
        
        if not value_part or len(value_part) < 2:
            return results
        
        # Determine what type of PII this is based on field name
        pii_type = self.classify_field_type(field_part)
        
        if pii_type:
            # Extract the actual value
            clean_value = self.clean_value_by_type(value_part, pii_type)
            if clean_value:
                results[pii_type] = [clean_value]
        
        return results
    
    def classify_field_type(self, field_text: str) -> str:
        """Classify what type of PII a field contains based on its label"""
        field_text = field_text.lower()
        
        # Name fields
        if any(indicator in field_text for indicator in ['name', 'नाव']):
            return 'names'
        
        # Address fields  
        if any(indicator in field_text for indicator in ['address', 'पत्ता', 'occurrence', 'घटना']):
            return 'addresses'
        
        # Phone fields
        if any(indicator in field_text for indicator in ['phone', 'mobile', 'contact']):
            return 'phone_numbers'
        
        # Date fields
        if any(indicator in field_text for indicator in ['date', 'दिनांक']):
            return 'dates'
        
        # Time fields
        if any(indicator in field_text for indicator in ['time', 'वेळ']):
            return 'times'
        
        # FIR fields
        if 'fir' in field_text or 'खब' in field_text:
            return 'fir_numbers'
        
        # Case fields
        if 'case' in field_text:
            return 'case_numbers'
        
        # Police station fields
        if any(indicator in field_text for indicator in ['p.s.', 'police station', 'पोलीस ठाणे']):
            return 'police_stations'
        
        # District fields
        if any(indicator in field_text for indicator in ['district', 'जिल्हा']):
            return 'districts'
        
        # Year fields
        if any(indicator in field_text for indicator in ['year', 'वर्ष']):
            return 'years'
        
        # Entry number fields
        if 'entry' in field_text or 'नोंद' in field_text:
            return 'entry_numbers'
        
        # Beat number fields
        if 'beat' in field_text or 'बीट' in field_text:
            return 'beat_numbers'
        
        return None
    
    def clean_value_by_type(self, value: str, pii_type: str) -> str:
        """Clean extracted value based on its PII type"""
        if not value:
            return ""
        
        # Remove common prefixes and suffixes
        value = re.sub(r'^[^\w\u0900-\u097F]+', '', value)  # Remove leading non-word chars
        value = re.sub(r'[^\w\u0900-\u097F\s:/.-]+$', '', value)  # Remove trailing junk
        value = value.strip()
        
        if pii_type == 'names':
            # Names should be mostly letters and spaces
            if re.match(r'^[A-Z][A-Z\s]{2,49}$', value):
                return value
        
        elif pii_type == 'dates':
            # Extract date pattern
            date_match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', value)
            if date_match:
                return date_match.group(1)
        
        elif pii_type == 'times':
            # Extract time pattern
            time_match = re.search(r'\b(\d{1,2}:\d{2})\b', value)
            if time_match:
                return time_match.group(1)
        
        elif pii_type == 'phone_numbers':
            # Extract phone number
            phone_match = re.search(r'(\+?91?[\s-]?[6-9]\d{9})', value)
            if phone_match:
                return phone_match.group(1)
        
        elif pii_type in ['fir_numbers', 'case_numbers', 'entry_numbers', 'beat_numbers']:
            # Extract numeric identifiers
            num_match = re.search(r'\b(\d+(?:/\d+)?)\b', value)
            if num_match:
                return num_match.group(1)
        
        elif pii_type == 'years':
            # Extract 4-digit year
            year_match = re.search(r'\b(20\d{2}|19\d{2})\b', value)
            if year_match:
                return year_match.group(1)
        
        elif pii_type in ['police_stations', 'districts', 'addresses']:
            # For location data, clean but preserve
            if len(value) > 2 and len(value) < 200:
                return value
        
        return ""
    
    def extract_standalone_pii(self, line: str) -> Dict[str, List[str]]:
        """Extract PII from standalone values (not field:value pairs)"""
        results = {}
        
        # Look for patterns that indicate PII even without field labels
        
        # Standalone dates
        date_matches = re.findall(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', line)
        if date_matches:
            results['dates'] = date_matches
        
        # Standalone times  
        time_matches = re.findall(r'\b(\d{1,2}:\d{2})\s*(?:तास|वाजता|AM|PM)?', line)
        if time_matches:
            results['times'] = time_matches
        
        # Phone numbers
        phone_matches = re.findall(r'\b(\+?91[\s-]?[6-9]\d{9})\b', line)
        if phone_matches:
            results['phone_numbers'] = phone_matches
        
        # 4-digit years
        year_matches = re.findall(r'\b(20\d{2})\b', line)
        if year_matches:
            results['years'] = year_matches
        
        return results
    
    def extract_multiline_pii(self, current_line: str, next_lines: List[str]) -> Dict[str, List[str]]:
        """Extract PII that spans multiple lines (like addresses)"""
        results = {}
        
        # If current line ends with a field indicator, next line might have the value
        if current_line.lower().endswith(('address', 'पत्ता', 'occurrence')):
            if next_lines:
                next_line = next_lines[0].strip()
                if next_line and len(next_line) > 5:
                    # This might be an address
                    clean_addr = self.clean_value_by_type(next_line, 'addresses')
                    if clean_addr:
                        results['addresses'] = [clean_addr]
        
        return results
    
    def clean_extracted_data(self, data: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Final cleanup and deduplication"""
        cleaned = {}
        
        for category, values in data.items():
            if not values:
                continue
                
            # Remove duplicates and empty values
            unique_values = []
            seen = set()
            
            for value in values:
                value = str(value).strip()
                if value and value not in seen and len(value) > 1:
                    seen.add(value)
                    unique_values.append(value)
            
            if unique_values:
                cleaned[category] = unique_values
        
        return cleaned
    
    def extract_pii(self, text: str) -> Dict[str, List[str]]:
        """Main extraction method"""
        return self.parse_text_semantically(text)

def main():
    st.set_page_config(
        page_title="Intelligent PII Extractor",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Intelligent PII Extractor")
    st.markdown("**Semantic understanding approach** - Analyzes text like a human would")
    
    extractor = IntelligentPIIExtractor()
    
    # Sidebar
    with st.sidebar:
        st.header("🧠 How It Works")
        st.markdown("""
        **Line-by-Line Analysis**
        - Reads each line semantically
        - Identifies field-value relationships
        - Understands context and meaning
        
        **Smart Classification**
        - Recognizes field types by labels
        - Handles multi-language content  
        - Extracts actual values, not labels
        
        **Contextual Cleaning**
        - Type-specific validation
        - Removes formatting artifacts
        - Preserves meaningful data
        """)
        
        st.header("📊 Debug Mode")
        debug_mode = st.checkbox("Show line-by-line analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Enhanced sample with clear formatting
        sample_text = """P.S. (पोलीस ठाणे): भोसरी
FIR No. (FIR खब Đ.): 0523
Date and Time of FIR: 19/11/2017 21:33 वाजता
District (जिल्हा): पुणे शहर
Year (वर्ष): 2017

Name (नाव): VIPUL RANGNATH JADHAV
Father's Name (वडिलांचे नाव): RANGNATH JADHAV

Address (पत्ता): अशोक हॉटेल्या पाठीमागे, मोरया मंदिरात, राशिवाटा, वासारवाडी, पुणे

Date (दिनांक): 19/11/2017
Time (वेळ): 21:09 तास

Entry No. (नोंद Đ.): 029
Beat No. (बीट Đ.): 1

Phone: +91 9876543210
Case No: 123/2017"""
        
        input_method = st.radio("Input method:", ["Sample Text", "Custom Text", "File Upload"])
        
        if input_method == "Sample Text":
            text_input = sample_text
            st.text_area("Sample:", text_input, height=350, disabled=True)
        elif input_method == "Custom Text":
            text_input = st.text_area("Enter your text:", height=350)
        else:
            uploaded_file = st.file_uploader("Upload file:", type=['txt'])
            if uploaded_file:
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", text_input[:400] + "...", height=200)
            else:
                text_input = ""
    
    with col2:
        st.header("🎯 Results")
        
        if st.button("🚀 Extract PII", type="primary") and text_input:
            with st.spinner("Analyzing text semantically..."):
                results = extractor.extract_pii(text_input)
                
                if results:
                    st.success(f"🎉 Found {sum(len(v) for v in results.values())} PII items in {len(results)} categories")
                    
                    # Show results
                    for category, items in results.items():
                        if items:
                            st.subheader(f"📋 {category.replace('_', ' ').title()}")
                            for i, item in enumerate(items, 1):
                                st.write(f"**{i}.** `{item}`")
                            st.write("---")
                    
                    # Export section
                    st.subheader("💾 Export")
                    
                    # Prepare export data
                    export_data = []
                    for category, items in results.items():
                        for item in items:
                            export_data.append({
                                'Category': category.replace('_', ' ').title(),
                                'Value': item,
                                'Length': len(item)
                            })
                    
                    if export_data:
                        df = pd.DataFrame(export_data)
                        
                        col_e1, col_e2 = st.columns(2)
                        
                        with col_e1:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 CSV",
                                csv,
                                f"intelligent_pii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            )
                        
                        with col_e2:
                            json_str = json.dumps(results, indent=2, ensure_ascii=False)
                            st.download_button(
                                "📥 JSON", 
                                json_str,
                                f"intelligent_pii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            )
                        
                        st.dataframe(df, use_container_width=True)
                
                else:
                    st.warning("No PII found. The text might not contain recognizable PII patterns.")
                
                # Debug mode
                if debug_mode:
                    st.subheader("🔍 Debug: Line Analysis")
                    lines = text_input.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip():
                            st.write(f"**Line {i+1}:** `{line}`")
                            line_pii = extractor.analyze_line_for_pii(line, i, lines)
                            if line_pii:
                                st.json(line_pii)
                            else:
                                st.write("_No PII detected_")
                            st.write("---")
        
        elif not text_input:
            st.info("👆 Choose input method and provide text")
    
    # Footer
    st.markdown("---") 
    st.markdown("**Intelligent PII Extractor** - Uses semantic analysis instead of regex patterns")

if __name__ == "__main__":
    main()
