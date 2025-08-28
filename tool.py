import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from elasticsearch import Elasticsearch
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MultilingualPIIExtractor:
    def __init__(self):
        self.setup_models()
        self.pii_patterns = self.define_pii_patterns()
        
    def setup_models(self):
        """Initialize multilingual models for PII detection"""
        try:
            # Load multilingual NER model from HuggingFace
            model_name = "microsoft/xlm-roberta-large-finetuned-conll03-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner", 
                model=model_name, 
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            
            # Try to load spaCy models
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp_en = None
                
            # For Hindi, we'll use regex patterns as spaCy Hindi model might not be available
            self.nlp_hi = None
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            self.ner_pipeline = None
            self.nlp_en = None
            self.nlp_hi = None
    
    def define_pii_patterns(self) -> Dict[str, List[str]]:
        """Define regex patterns for various PII types"""
        return {
            'phone_numbers': [
                r'\+?91[-\s]?\d{10}',  # Indian phone numbers
                r'\d{10}',  # 10 digit numbers
                r'\+\d{1,3}[-\s]?\d{6,14}',  # International numbers
                r'\d{3}[-\s]?\d{3}[-\s]?\d{4}'  # Standard format
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'pan_card': [
                r'[A-Z]{5}\d{4}[A-Z]',  # PAN card pattern
                r'[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]'  # More specific PAN
            ],
            'aadhaar': [
                r'\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # Aadhaar number
                r'\d{12}'  # 12 digit number
            ],
            'fir_number': [
                r'FIR\s*No\.?\s*:?\s*(\d+)',
                r'FIR\s*(\d+)',
                r'प्राथमिकी\s*संख्या\s*:?\s*(\d+)',
                r'एफआईआर\s*संख्या\s*:?\s*(\d+)'
            ],
            'case_number': [
                r'Case\s*No\.?\s*:?\s*([A-Z0-9/-]+)',
                r'मामला\s*संख्या\s*:?\s*([A-Z0-9/-]+)'
            ],
            'dates': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{1,2}\s+\w+\s+\d{2,4}',
                r'\d{2,4}-\d{1,2}-\d{1,2}'
            ],
            'times': [
                r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?',
                r'\d{1,2}\.\d{2}\s*(?:hours?|बजे)'
            ],
            'addresses': [
                r'(?:Village|Gram|गाँव|ग्राम)\s*:?\s*([^,\n]+)',
                r'(?:Police Station|P\.S\.|थाना)\s*:?\s*([^,\n]+)',
                r'(?:District|जिला)\s*:?\s*([^,\n]+)',
                r'(?:State|राज्य)\s*:?\s*([^,\n]+)',
                r'PIN\s*:?\s*(\d{6})',
                r'पिन\s*:?\s*(\d{6})'
            ],
            'names': [
                # Basic name patterns
                r'(?:Name|नाम|nama)\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Father|पिता|पिताजी|बाप)\s*(?:Name|नाम|का नाम)?\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Mother|माता|माताजी|माँ)\s*(?:Name|नाम|का नाम)?\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Son|पुत्र|Daughter|पुत्री|बेटा|बेटी)\s*(?:of|का|की)?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Wife|पत्नी|पत्नि)\s*(?:of|का|की)?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Husband|पति)\s*(?:Name|नाम)?\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                # Address patterns that often contain names
                r'(?:S/o|D/o|W/o)\s+([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:श्री|श्रीमती|कुमार|कुमारी)\s+([A-Za-z\u0900-\u097F\s]{2,50})',
                # Complainant patterns
                r'(?:Complainant|शिकायतकर्ता|आवेदक)\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Accused|आरोपी|अभियुक्त)\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                r'(?:Witness|गवाह|साक्षी)\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,50})',
                # Indian name patterns
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',  # English names
                r'\b([\u0900-\u097F]+(?:\s+[\u0900-\u097F]+){0,3})\b',  # Hindi names
                # Age-based patterns (names often appear with age)
                r'([A-Za-z\u0900-\u097F\s]{2,30})\s*(?:age|उम्र|आयु)\s*:?\s*\d+',
                r'(?:age|उम्र|आयु)\s*:?\s*\d+\s*(?:years?|साल|वर्ष)?\s*([A-Za-z\u0900-\u097F\s]{2,30})'
            ]
        }
    
    def extract_with_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract PII using regex patterns"""
        results = {}
        
        for pii_type, patterns in self.pii_patterns.items():
            matches = set()
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
                if found:
                    if isinstance(found[0], tuple):
                        matches.update([match for match in found if match])
                    else:
                        matches.update(found)
            
            # Clean and filter matches
            cleaned_matches = []
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(str(m) for m in match if m)
                
                match = str(match).strip()
                if len(match) > 1 and not match.isspace():
                    # Additional validation for specific types
                    if pii_type == 'aadhaar' and len(re.sub(r'[-\s]', '', match)) != 12:
                        continue
                    if pii_type == 'pan_card' and len(match) != 10:
                        continue
                    if pii_type == 'phone_numbers' and len(re.sub(r'[-\s+]', '', match)) < 10:
                        continue
                    
                    cleaned_matches.append(match)
            
            results[pii_type] = list(set(cleaned_matches))
        
        return results
    
    def extract_with_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract PII using NER models"""
        results = {'persons': [], 'locations': [], 'organizations': []}
        
        if not self.ner_pipeline:
            return results
        
        try:
            # Split text into chunks to avoid token limits
            chunks = [text[i:i+400] for i in range(0, len(text), 400)]
            
            for chunk in chunks:
                if len(chunk.strip()) < 5:
                    continue
                    
                entities = self.ner_pipeline(chunk)
                for entity in entities:
                    label = entity['entity_group'].upper()
                    word = entity['word'].strip()
                    
                    if len(word) > 1 and not word.isspace():
                        if label in ['PER', 'PERSON']:
                            results['persons'].append(word)
                        elif label in ['LOC', 'LOCATION', 'GPE']:
                            results['locations'].append(word)
                        elif label in ['ORG', 'ORGANIZATION']:
                            results['organizations'].append(word)
            
            # Also use spaCy for better name extraction if available
            if self.nlp_en:
                doc = self.nlp_en(text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        results['persons'].append(ent.text.strip())
            
            # Remove duplicates
            for key in results:
                results[key] = list(set(results[key]))
                
        except Exception as e:
            st.warning(f"NER extraction failed: {e}")
        
        return results
    
    def clean_and_validate_names(self, names: List[str]) -> List[str]:
        """Clean and validate extracted names"""
        valid_names = []
        
        # Common words to exclude (not names)
        exclude_words = {
            'police', 'station', 'thane', 'district', 'state', 'date', 'time', 'case', 'fir',
            'पुलिस', 'थाना', 'जिला', 'राज्य', 'दिनांक', 'समय', 'मामला', 'एफआईआर',
            'section', 'act', 'code', 'court', 'judge', 'lawyer', 'advocate',
            'धारा', 'अधिनियम', 'न्यायालय', 'न्यायाधीश', 'वकील', 'अधिवक्ता'
        }
        
        for name in names:
            name = name.strip()
            if not name or len(name) < 2:
                continue
            
            # Remove common prefixes/suffixes
            name = re.sub(r'^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|श्री|श्रीमती)\s*', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s*(?:ji|जी|saheb|साहब|sir|madam)
    def extract_pii(self, text: str) -> Dict[str, Any]:
        """Main PII extraction function"""
        if not text or len(text.strip()) < 5:
            return {}
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        # Keep more characters for names - don't over-clean
        cleaned_text = re.sub(r'[^\w\s@./:-\u0900-\u097F]', ' ', text)
        
        # Extract using regex patterns
        regex_results = self.extract_with_regex(cleaned_text)
        
        # Extract using NER
        ner_results = self.extract_with_ner(cleaned_text)
        
        # Advanced name extraction
        extracted_names = self.extract_names_advanced(text)
        if extracted_names:
            if 'names' in regex_results:
                regex_results['names'].extend(extracted_names)
            else:
                regex_results['names'] = extracted_names
        
        # Combine NER persons with names
        if 'persons' in ner_results and ner_results['persons']:
            if 'names' in regex_results:
                regex_results['names'].extend(ner_results['persons'])
            else:
                regex_results['names'] = ner_results['persons']
        
        # Clean up names list
        if 'names' in regex_results:
            regex_results['names'] = self.clean_and_validate_names(regex_results['names'])
        
        # Combine results (excluding persons as it's now merged with names)
        all_results = {**regex_results}
        for key in ['locations', 'organizations']:
            if key in ner_results:
                if key in all_results:
                    all_results[key].extend(ner_results[key])
                else:
                    all_results[key] = ner_results[key]
        
        # Remove empty results and duplicates
        final_results = {}
        for k, v in all_results.items():
            if v:
                # Remove duplicates while preserving order
                seen = set()
                unique_values = []
                for item in v:
                    if item not in seen:
                        seen.add(item)
                        unique_values.append(item)
                final_results[k] = unique_values
        
        return final_results
    
    def format_results(self, pii_data: Dict[str, Any]) -> pd.DataFrame:
        """Format PII extraction results as DataFrame"""
        rows = []
        for pii_type, values in pii_data.items():
            for value in values:
                rows.append({
                    'PII_Type': pii_type.replace('_', ' ').title(),
                    'Value': value,
                    'Confidence': 'High' if pii_type in ['email', 'phone_numbers', 'pan_card', 'aadhaar'] else 'Medium',
                    'Category': self.get_pii_category(pii_type)
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['PII_Type', 'Value', 'Confidence', 'Category'])
    
    def get_pii_category(self, pii_type: str) -> str:
        """Categorize PII types"""
        categories = {
            'phone_numbers': 'Contact',
            'email': 'Contact',
            'names': 'Identity',
            'persons': 'Identity',
            'pan_card': 'Government ID',
            'aadhaar': 'Government ID',
            'fir_number': 'Legal Document',
            'case_number': 'Legal Document',
            'addresses': 'Location',
            'locations': 'Location',
            'dates': 'Temporal',
            'times': 'Temporal',
            'organizations': 'Organization'
        }
        return categories.get(pii_type, 'Other')

def connect_elasticsearch(host: str, port: int, username: str = None, password: str = None) -> Elasticsearch:
    """Connect to Elasticsearch"""
    try:
        if username and password:
            es = Elasticsearch(
                [{'host': host, 'port': port}],
                http_auth=(username, password),
                timeout=30
            )
        else:
            es = Elasticsearch([{'host': host, 'port': port}], timeout=30)
        
        if es.ping():
            return es
        else:
            st.error("Could not connect to Elasticsearch")
            return None
    except Exception as e:
        st.error(f"Elasticsearch connection error: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Multilingual PII Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multilingual PII Extractor for Legal Documents")
    st.markdown("Extract Personally Identifiable Information from Hindi/English legal texts")
    
    # Initialize PII extractor
    if 'pii_extractor' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.pii_extractor = MultilingualPIIExtractor()
    
    extractor = st.session_state.pii_extractor
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Text Input", "Elasticsearch", "File Upload"]
    )
    
    if data_source == "Elasticsearch":
        st.sidebar.subheader("Elasticsearch Settings")
        es_host = st.sidebar.text_input("Host", value="localhost")
        es_port = st.sidebar.number_input("Port", value=9200, min_value=1, max_value=65535)
        es_username = st.sidebar.text_input("Username (optional)")
        es_password = st.sidebar.text_input("Password (optional)", type="password")
        es_index = st.sidebar.text_input("Index Name")
        es_query = st.sidebar.text_area("Query (JSON)", value='{"match_all": {}}')
        
        if st.sidebar.button("Connect & Extract"):
            es = connect_elasticsearch(es_host, es_port, es_username, es_password)
            if es and es_index:
                try:
                    query = json.loads(es_query)
                    response = es.search(index=es_index, body={"query": query}, size=100)
                    
                    all_results = []
                    progress_bar = st.progress(0)
                    
                    for i, hit in enumerate(response['hits']['hits']):
                        # Extract text from all fields
                        text_content = ' '.join([str(v) for v in hit['_source'].values() if isinstance(v, (str, int, float))])
                        
                        pii_data = extractor.extract_pii(text_content)
                        if pii_data:
                            df = extractor.format_results(pii_data)
                            df['Document_ID'] = hit['_id']
                            all_results.append(df)
                        
                        progress_bar.progress((i + 1) / len(response['hits']['hits']))
                    
                    if all_results:
                        final_df = pd.concat(all_results, ignore_index=True)
                        st.success(f"Extracted PII from {len(response['hits']['hits'])} documents")
                        st.dataframe(final_df)
                        
                        # Download results
                        csv = final_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the documents")
                        
                except Exception as e:
                    st.error(f"Error processing Elasticsearch data: {e}")
    
    elif data_source == "File Upload":
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'json'])
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                    pii_data = extractor.extract_pii(text)
                    
                    if pii_data:
                        df = extractor.format_results(pii_data)
                        st.success("PII extraction completed!")
                        st.dataframe(df)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the document")
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Text Input
        st.subheader("Text Input")
        
        # Sample text button
        if st.button("Load Sample Text"):
            sample_text = """P.S. (Police Thane): Bhosari 
FIR No. (एफआईआर संख्या): 0523 
Date and Time of FIR: 19/11/2017 at 21:33 
District (जिला): Pune City 
Year: 2017

Complainant: राम कुमार शर्मा
Father's Name: श्यामलाल शर्मा  
Mother's Name: सीता देवी
Age: 35 years

Accused: अमित पटेल S/o राजेश पटेल
Witness: प्रिया शर्मा D/o मोहन शर्मा

Phone: +91-9876543210
Email: ram.kumar@email.com
Address: शोभा हेट सिया सड़क, मोया मार्केट, अशिता, असारावाडी, Pune, PIN: 411001
PAN Card: ABCDE1234F
Aadhaar: 1234-5678-9012

Against: संदीप कुमार W/o रेखा कुमार
By: Inspector Rajesh Singh"""
            st.session_state.input_text = sample_text
        
        text_input = st.text_area(
            "Enter text for PII extraction:",
            height=300,
            value=st.session_state.get('input_text', ''),
            placeholder="Paste your legal document text here..."
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Extract PII", type="primary"):
                if text_input.strip():
                    with st.spinner("Extracting PII..."):
                        pii_data = extractor.extract_pii(text_input)
                    
                    if pii_data:
                        df = extractor.format_results(pii_data)
                        st.success("PII extraction completed!")
                        
                        # Display results
                        st.subheader("Extracted PII")
                        st.dataframe(df)
                        
                        # Statistics
                        st.subheader("Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total PII Items", len(df))
                        with col2:
                            st.metric("PII Types", df['PII_Type'].nunique())
                        with col3:
                            st.metric("High Confidence Items", len(df[df['Confidence'] == 'High']))
                        
                        # Category breakdown
                        if not df.empty:
                            st.subheader("PII by Category")
                            category_counts = df['Category'].value_counts()
                            st.bar_chart(category_counts)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the text")
                else:
                    st.warning("Please enter some text")
        
        with col2:
            if st.button("Clear Text"):
                st.session_state.input_text = ""
                st.experimental_rerun()
    
    # Information section
    with st.expander("ℹ️ About PII Types"):
        st.markdown("""
        **Supported PII Types:**
        - **Contact**: Phone numbers, Email addresses
        - **Identity**: Names, Person entities
        - **Government ID**: PAN Card, Aadhaar numbers
        - **Legal Document**: FIR numbers, Case numbers
        - **Location**: Addresses, Places, Geographic entities
        - **Temporal**: Dates, Times
        - **Organization**: Organization names
        
        **Languages Supported:**
        - English
        - Hindi (हिंदी)
        - Mixed language text
        
        **Special Features:**
        - OCR text processing
        - Legal document patterns
        - Indian government ID formats
        - Multilingual named entity recognition
        """)

if __name__ == "__main__":
    main(), '', name, flags=re.IGNORECASE)
            
            # Skip if it's a common word
            if name.lower() in exclude_words:
                continue
            
            # Skip if it's all numbers or special characters
            if re.match(r'^[\d\W]+
        """Main PII extraction function"""
        if not text or len(text.strip()) < 5:
            return {}
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s@./:-]', ' ', text)
        
        # Extract using regex patterns
        regex_results = self.extract_with_regex(text)
        
        # Extract using NER
        ner_results = self.extract_with_ner(text)
        
        # Combine results
        all_results = {**regex_results, **ner_results}
        
        # Remove empty results
        final_results = {k: v for k, v in all_results.items() if v}
        
        return final_results
    
    def format_results(self, pii_data: Dict[str, Any]) -> pd.DataFrame:
        """Format PII extraction results as DataFrame"""
        rows = []
        for pii_type, values in pii_data.items():
            for value in values:
                rows.append({
                    'PII_Type': pii_type.replace('_', ' ').title(),
                    'Value': value,
                    'Confidence': 'High' if pii_type in ['email', 'phone_numbers', 'pan_card', 'aadhaar'] else 'Medium',
                    'Category': self.get_pii_category(pii_type)
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['PII_Type', 'Value', 'Confidence', 'Category'])
    
    def get_pii_category(self, pii_type: str) -> str:
        """Categorize PII types"""
        categories = {
            'phone_numbers': 'Contact',
            'email': 'Contact',
            'names': 'Identity',
            'persons': 'Identity',
            'pan_card': 'Government ID',
            'aadhaar': 'Government ID',
            'fir_number': 'Legal Document',
            'case_number': 'Legal Document',
            'addresses': 'Location',
            'locations': 'Location',
            'dates': 'Temporal',
            'times': 'Temporal',
            'organizations': 'Organization'
        }
        return categories.get(pii_type, 'Other')

def connect_elasticsearch(host: str, port: int, username: str = None, password: str = None) -> Elasticsearch:
    """Connect to Elasticsearch"""
    try:
        if username and password:
            es = Elasticsearch(
                [{'host': host, 'port': port}],
                http_auth=(username, password),
                timeout=30
            )
        else:
            es = Elasticsearch([{'host': host, 'port': port}], timeout=30)
        
        if es.ping():
            return es
        else:
            st.error("Could not connect to Elasticsearch")
            return None
    except Exception as e:
        st.error(f"Elasticsearch connection error: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Multilingual PII Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multilingual PII Extractor for Legal Documents")
    st.markdown("Extract Personally Identifiable Information from Hindi/English legal texts")
    
    # Initialize PII extractor
    if 'pii_extractor' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.pii_extractor = MultilingualPIIExtractor()
    
    extractor = st.session_state.pii_extractor
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Text Input", "Elasticsearch", "File Upload"]
    )
    
    if data_source == "Elasticsearch":
        st.sidebar.subheader("Elasticsearch Settings")
        es_host = st.sidebar.text_input("Host", value="localhost")
        es_port = st.sidebar.number_input("Port", value=9200, min_value=1, max_value=65535)
        es_username = st.sidebar.text_input("Username (optional)")
        es_password = st.sidebar.text_input("Password (optional)", type="password")
        es_index = st.sidebar.text_input("Index Name")
        es_query = st.sidebar.text_area("Query (JSON)", value='{"match_all": {}}')
        
        if st.sidebar.button("Connect & Extract"):
            es = connect_elasticsearch(es_host, es_port, es_username, es_password)
            if es and es_index:
                try:
                    query = json.loads(es_query)
                    response = es.search(index=es_index, body={"query": query}, size=100)
                    
                    all_results = []
                    progress_bar = st.progress(0)
                    
                    for i, hit in enumerate(response['hits']['hits']):
                        # Extract text from all fields
                        text_content = ' '.join([str(v) for v in hit['_source'].values() if isinstance(v, (str, int, float))])
                        
                        pii_data = extractor.extract_pii(text_content)
                        if pii_data:
                            df = extractor.format_results(pii_data)
                            df['Document_ID'] = hit['_id']
                            all_results.append(df)
                        
                        progress_bar.progress((i + 1) / len(response['hits']['hits']))
                    
                    if all_results:
                        final_df = pd.concat(all_results, ignore_index=True)
                        st.success(f"Extracted PII from {len(response['hits']['hits'])} documents")
                        st.dataframe(final_df)
                        
                        # Download results
                        csv = final_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the documents")
                        
                except Exception as e:
                    st.error(f"Error processing Elasticsearch data: {e}")
    
    elif data_source == "File Upload":
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'json'])
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                    pii_data = extractor.extract_pii(text)
                    
                    if pii_data:
                        df = extractor.format_results(pii_data)
                        st.success("PII extraction completed!")
                        st.dataframe(df)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the document")
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Text Input
        st.subheader("Text Input")
        
        # Sample text button
        if st.button("Load Sample Text"):
            sample_text = """P.S. (Police Thane): Bhosari 
FIR No. (एफआईआर संख्या): 0523 
Date and Time of FIR: 19/11/2017 at 21:33 
District (जिला): Pune City 
Year: 2017
Name: राम कुमार शर्मा
Father's Name: श्यामलाल शर्मा  
Phone: +91-9876543210
Email: ram.kumar@email.com
Address: शोभा हेट सिया सड़क, मोया मार्केट, अशिता, असारावाडी, Pune, PIN: 411001
PAN Card: ABCDE1234F
Aadhaar: 1234-5678-9012"""
            st.session_state.input_text = sample_text
        
        text_input = st.text_area(
            "Enter text for PII extraction:",
            height=300,
            value=st.session_state.get('input_text', ''),
            placeholder="Paste your legal document text here..."
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Extract PII", type="primary"):
                if text_input.strip():
                    with st.spinner("Extracting PII..."):
                        pii_data = extractor.extract_pii(text_input)
                    
                    if pii_data:
                        df = extractor.format_results(pii_data)
                        st.success("PII extraction completed!")
                        
                        # Display results
                        st.subheader("Extracted PII")
                        st.dataframe(df)
                        
                        # Statistics
                        st.subheader("Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total PII Items", len(df))
                        with col2:
                            st.metric("PII Types", df['PII_Type'].nunique())
                        with col3:
                            st.metric("High Confidence Items", len(df[df['Confidence'] == 'High']))
                        
                        # Category breakdown
                        if not df.empty:
                            st.subheader("PII by Category")
                            category_counts = df['Category'].value_counts()
                            st.bar_chart(category_counts)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the text")
                else:
                    st.warning("Please enter some text")
        
        with col2:
            if st.button("Clear Text"):
                st.session_state.input_text = ""
                st.experimental_rerun()
    
    # Information section
    with st.expander("ℹ️ About PII Types"):
        st.markdown("""
        **Supported PII Types:**
        - **Contact**: Phone numbers, Email addresses
        - **Identity**: Names, Person entities
        - **Government ID**: PAN Card, Aadhaar numbers
        - **Legal Document**: FIR numbers, Case numbers
        - **Location**: Addresses, Places, Geographic entities
        - **Temporal**: Dates, Times
        - **Organization**: Organization names
        
        **Languages Supported:**
        - English
        - Hindi (हिंदी)
        - Mixed language text
        
        **Special Features:**
        - OCR text processing
        - Legal document patterns
        - Indian government ID formats
        - Multilingual named entity recognition
        """)

if __name__ == "__main__":
    main(), name):
                continue
            
            # Skip very long names (likely not actual names)
            if len(name) > 50:
                continue
            
            # Skip single characters
            if len(name.replace(' ', '')) < 2:
                continue
            
            valid_names.append(name.title())
        
        return list(set(valid_names))

    def extract_names_advanced(self, text: str) -> List[str]:
        """Advanced name extraction with multiple techniques"""
        all_names = []
        
        # Method 1: Regex patterns
        regex_results = self.extract_with_regex(text)
        if 'names' in regex_results:
            all_names.extend(regex_results['names'])
        
        # Method 2: Look for capitalized words (likely names)
        # Find sequences of capitalized words
        cap_pattern = r'\b[A-Z\u0900-\u097F][a-z\u0900-\u097F]+(?:\s+[A-Z\u0900-\u097F][a-z\u0900-\u097F]+)*\b'
        potential_names = re.findall(cap_pattern, text)
        all_names.extend(potential_names)
        
        # Method 3: Context-based extraction
        # Look for patterns like "Name of person: XYZ"
        context_patterns = [
            r'(?:informant|complainant|accused|witness|victim|applicant|respondent)\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,40})',
            r'(?:सूचनाकर्ता|शिकायतकर्ता|आरोपी|गवाह|पीड़ित|आवेदक)\s*:?\s*([A-Za-z\u0900-\u097F\s]{2,40})',
            r'(?:against|के विरुद्ध)\s+([A-Za-z\u0900-\u097F\s]{2,40})',
            r'(?:by|द्वारा)\s+([A-Za-z\u0900-\u097F\s]{2,40})'
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.UNICODE)
            all_names.extend(matches)
        
        # Clean and validate names
        return self.clean_and_validate_names(all_names)
        """Main PII extraction function"""
        if not text or len(text.strip()) < 5:
            return {}
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s@./:-]', ' ', text)
        
        # Extract using regex patterns
        regex_results = self.extract_with_regex(text)
        
        # Extract using NER
        ner_results = self.extract_with_ner(text)
        
        # Combine results
        all_results = {**regex_results, **ner_results}
        
        # Remove empty results
        final_results = {k: v for k, v in all_results.items() if v}
        
        return final_results
    
    def format_results(self, pii_data: Dict[str, Any]) -> pd.DataFrame:
        """Format PII extraction results as DataFrame"""
        rows = []
        for pii_type, values in pii_data.items():
            for value in values:
                rows.append({
                    'PII_Type': pii_type.replace('_', ' ').title(),
                    'Value': value,
                    'Confidence': 'High' if pii_type in ['email', 'phone_numbers', 'pan_card', 'aadhaar'] else 'Medium',
                    'Category': self.get_pii_category(pii_type)
                })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['PII_Type', 'Value', 'Confidence', 'Category'])
    
    def get_pii_category(self, pii_type: str) -> str:
        """Categorize PII types"""
        categories = {
            'phone_numbers': 'Contact',
            'email': 'Contact',
            'names': 'Identity',
            'persons': 'Identity',
            'pan_card': 'Government ID',
            'aadhaar': 'Government ID',
            'fir_number': 'Legal Document',
            'case_number': 'Legal Document',
            'addresses': 'Location',
            'locations': 'Location',
            'dates': 'Temporal',
            'times': 'Temporal',
            'organizations': 'Organization'
        }
        return categories.get(pii_type, 'Other')

def connect_elasticsearch(host: str, port: int, username: str = None, password: str = None) -> Elasticsearch:
    """Connect to Elasticsearch"""
    try:
        if username and password:
            es = Elasticsearch(
                [{'host': host, 'port': port}],
                http_auth=(username, password),
                timeout=30
            )
        else:
            es = Elasticsearch([{'host': host, 'port': port}], timeout=30)
        
        if es.ping():
            return es
        else:
            st.error("Could not connect to Elasticsearch")
            return None
    except Exception as e:
        st.error(f"Elasticsearch connection error: {e}")
        return None

def main():
    st.set_page_config(
        page_title="Multilingual PII Extractor",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multilingual PII Extractor for Legal Documents")
    st.markdown("Extract Personally Identifiable Information from Hindi/English legal texts")
    
    # Initialize PII extractor
    if 'pii_extractor' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.pii_extractor = MultilingualPIIExtractor()
    
    extractor = st.session_state.pii_extractor
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Text Input", "Elasticsearch", "File Upload"]
    )
    
    if data_source == "Elasticsearch":
        st.sidebar.subheader("Elasticsearch Settings")
        es_host = st.sidebar.text_input("Host", value="localhost")
        es_port = st.sidebar.number_input("Port", value=9200, min_value=1, max_value=65535)
        es_username = st.sidebar.text_input("Username (optional)")
        es_password = st.sidebar.text_input("Password (optional)", type="password")
        es_index = st.sidebar.text_input("Index Name")
        es_query = st.sidebar.text_area("Query (JSON)", value='{"match_all": {}}')
        
        if st.sidebar.button("Connect & Extract"):
            es = connect_elasticsearch(es_host, es_port, es_username, es_password)
            if es and es_index:
                try:
                    query = json.loads(es_query)
                    response = es.search(index=es_index, body={"query": query}, size=100)
                    
                    all_results = []
                    progress_bar = st.progress(0)
                    
                    for i, hit in enumerate(response['hits']['hits']):
                        # Extract text from all fields
                        text_content = ' '.join([str(v) for v in hit['_source'].values() if isinstance(v, (str, int, float))])
                        
                        pii_data = extractor.extract_pii(text_content)
                        if pii_data:
                            df = extractor.format_results(pii_data)
                            df['Document_ID'] = hit['_id']
                            all_results.append(df)
                        
                        progress_bar.progress((i + 1) / len(response['hits']['hits']))
                    
                    if all_results:
                        final_df = pd.concat(all_results, ignore_index=True)
                        st.success(f"Extracted PII from {len(response['hits']['hits'])} documents")
                        st.dataframe(final_df)
                        
                        # Download results
                        csv = final_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the documents")
                        
                except Exception as e:
                    st.error(f"Error processing Elasticsearch data: {e}")
    
    elif data_source == "File Upload":
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'json'])
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "text/plain":
                    text = str(uploaded_file.read(), "utf-8")
                    pii_data = extractor.extract_pii(text)
                    
                    if pii_data:
                        df = extractor.format_results(pii_data)
                        st.success("PII extraction completed!")
                        st.dataframe(df)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the document")
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    else:  # Text Input
        st.subheader("Text Input")
        
        # Sample text button
        if st.button("Load Sample Text"):
            sample_text = """P.S. (Police Thane): Bhosari 
FIR No. (एफआईआर संख्या): 0523 
Date and Time of FIR: 19/11/2017 at 21:33 
District (जिला): Pune City 
Year: 2017
Name: राम कुमार शर्मा
Father's Name: श्यामलाल शर्मा  
Phone: +91-9876543210
Email: ram.kumar@email.com
Address: शोभा हेट सिया सड़क, मोया मार्केट, अशिता, असारावाडी, Pune, PIN: 411001
PAN Card: ABCDE1234F
Aadhaar: 1234-5678-9012"""
            st.session_state.input_text = sample_text
        
        text_input = st.text_area(
            "Enter text for PII extraction:",
            height=300,
            value=st.session_state.get('input_text', ''),
            placeholder="Paste your legal document text here..."
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Extract PII", type="primary"):
                if text_input.strip():
                    with st.spinner("Extracting PII..."):
                        pii_data = extractor.extract_pii(text_input)
                    
                    if pii_data:
                        df = extractor.format_results(pii_data)
                        st.success("PII extraction completed!")
                        
                        # Display results
                        st.subheader("Extracted PII")
                        st.dataframe(df)
                        
                        # Statistics
                        st.subheader("Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total PII Items", len(df))
                        with col2:
                            st.metric("PII Types", df['PII_Type'].nunique())
                        with col3:
                            st.metric("High Confidence Items", len(df[df['Confidence'] == 'High']))
                        
                        # Category breakdown
                        if not df.empty:
                            st.subheader("PII by Category")
                            category_counts = df['Category'].value_counts()
                            st.bar_chart(category_counts)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No PII found in the text")
                else:
                    st.warning("Please enter some text")
        
        with col2:
            if st.button("Clear Text"):
                st.session_state.input_text = ""
                st.experimental_rerun()
    
    # Information section
    with st.expander("ℹ️ About PII Types"):
        st.markdown("""
        **Supported PII Types:**
        - **Contact**: Phone numbers, Email addresses
        - **Identity**: Names, Person entities
        - **Government ID**: PAN Card, Aadhaar numbers
        - **Legal Document**: FIR numbers, Case numbers
        - **Location**: Addresses, Places, Geographic entities
        - **Temporal**: Dates, Times
        - **Organization**: Organization names
        
        **Languages Supported:**
        - English
        - Hindi (हिंदी)
        - Mixed language text
        
        **Special Features:**
        - OCR text processing
        - Legal document patterns
        - Indian government ID formats
        - Multilingual named entity recognition
        """)

if __name__ == "__main__":
    main()
