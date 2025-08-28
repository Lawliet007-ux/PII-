import streamlit as st
import pandas as pd
import re
import json
from datetime import datetime
from typing import Dict, List, Set, Any
import spacy
from spacy import displacy
import warnings
warnings.filterwarnings('ignore')

# For Elasticsearch connection
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, AuthenticationException

# For advanced NER models
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# For phone number detection
import phonenumbers
from phonenumbers import geocoder, carrier

# For email validation
import email_validator

# Configuration
st.set_page_config(
    page_title="Multilingual PII Extraction Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PIIExtractor:
    def __init__(self):
        self.pii_patterns = self._load_patterns()
        self.hindi_ner_model = None
        self.english_ner_model = None
        self.multilingual_ner = None
        
    def _load_patterns(self) -> Dict[str, str]:
        """Load regex patterns for different PII types"""
        return {
            # Indian specific patterns
            'aadhaar': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'pan': r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
            'indian_mobile': r'\b(?:\+91|0)?[6-9]\d{9}\b',
            'gstin': r'\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b',
            'ifsc': r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
            'indian_passport': r'\b[A-Z]{1}\d{7}\b',
            'voter_id': r'\b[A-Z]{3}\d{7}\b',
            'driving_license': r'\b[A-Z]{2}[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{7}\b',
            
            # International patterns
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'credit_card': r'\b(?:\d{4}[\s-]?){3}\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'date_birth': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'pincode': r'\b\d{6}\b',
            'bank_account': r'\b\d{9,18}\b',
            
            # Address patterns (basic)
            'address_keywords': r'\b(?:house|flat|apartment|block|sector|phase|colony|nagar|road|street|lane|gali|marg|chowk|circle|square)\b',
            
            # Hindi name patterns (common prefixes/suffixes)
            'hindi_name_prefixes': r'\b(?:श्री|श्रीमती|कुमारी|डॉ|प्रो|मिस्टर|मिसेज)\b',
            'hindi_name_suffixes': r'\b(?:जी|साहब|साहिब|जी|महोदय|महोदया)\b'
        }
    
    @st.cache_resource
    def load_models(_self):
        """Load NER models"""
        try:
            # Load multilingual NER model (good for Hindi/English)
            _self.multilingual_ner = pipeline(
                "ner", 
                model="microsoft/DialoGPT-medium",  # Fallback model
                tokenizer="microsoft/DialoGPT-medium",
                aggregation_strategy="simple"
            )
            
            # Try to load specialized models
            try:
                # Hindi NER model
                _self.hindi_ner_model = pipeline(
                    "ner",
                    model="ai4bharat/indic-bert",
                    tokenizer="ai4bharat/indic-bert", 
                    aggregation_strategy="simple"
                )
            except:
                st.warning("Could not load Hindi NER model, using multilingual fallback")
            
            # English NER with spaCy
            try:
                import en_core_web_sm
                _self.english_nlp = en_core_web_sm.load()
            except:
                st.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                _self.english_nlp = None
                
        except Exception as e:
            st.error(f"Error loading NER models: {e}")
            _self.multilingual_ner = None
    
    def extract_regex_pii(self, text: str) -> Dict[str, List[str]]:
        """Extract PII using regex patterns"""
        extracted_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Remove duplicates while preserving order
                unique_matches = list(dict.fromkeys(matches))
                extracted_pii[pii_type] = unique_matches
        
        return extracted_pii
    
    def extract_ner_pii(self, text: str) -> Dict[str, List[str]]:
        """Extract PII using NER models"""
        extracted_pii = {'persons': [], 'organizations': [], 'locations': []}
        
        # Use English spaCy model
        if hasattr(self, 'english_nlp') and self.english_nlp:
            doc = self.english_nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    extracted_pii['persons'].append(ent.text)
                elif ent.label_ in ["ORG", "ORGANIZATION"]:
                    extracted_pii['organizations'].append(ent.text)
                elif ent.label_ in ["GPE", "LOC", "LOCATION"]:
                    extracted_pii['locations'].append(ent.text)
        
        # Use multilingual model as backup
        if self.multilingual_ner:
            try:
                results = self.multilingual_ner(text)
                for result in results:
                    if result['entity_group'] in ['PERSON', 'PER']:
                        extracted_pii['persons'].append(result['word'])
                    elif result['entity_group'] in ['ORG', 'ORGANIZATION']:
                        extracted_pii['organizations'].append(result['word'])
                    elif result['entity_group'] in ['LOC', 'LOCATION']:
                        extracted_pii['locations'].append(result['word'])
            except Exception as e:
                st.warning(f"NER extraction failed: {e}")
        
        # Remove duplicates
        for key in extracted_pii:
            extracted_pii[key] = list(dict.fromkeys(extracted_pii[key]))
            
        return extracted_pii
    
    def validate_phone_numbers(self, phone_numbers: List[str]) -> List[Dict]:
        """Validate and get details for phone numbers"""
        validated_phones = []
        
        for phone in phone_numbers:
            try:
                # Clean the phone number
                clean_phone = re.sub(r'[^\d+]', '', phone)
                
                # Parse with country code for India
                if not clean_phone.startswith('+'):
                    clean_phone = '+91' + clean_phone.lstrip('0')
                
                parsed = phonenumbers.parse(clean_phone, "IN")
                
                if phonenumbers.is_valid_number(parsed):
                    validated_phones.append({
                        'original': phone,
                        'formatted': phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL),
                        'country': geocoder.description_for_number(parsed, "en"),
                        'carrier': carrier.name_for_number(parsed, "en"),
                        'valid': True
                    })
            except:
                # If validation fails, still include as potential PII
                validated_phones.append({
                    'original': phone,
                    'valid': False
                })
        
        return validated_phones
    
    def validate_emails(self, emails: List[str]) -> List[Dict]:
        """Validate email addresses"""
        validated_emails = []
        
        for email in emails:
            try:
                valid_email = email_validator.validate_email(email)
                validated_emails.append({
                    'original': email,
                    'normalized': valid_email.email,
                    'valid': True
                })
            except:
                validated_emails.append({
                    'original': email,
                    'valid': False
                })
        
        return validated_emails
    
    def is_likely_pii(self, text: str, pii_type: str) -> bool:
        """Additional validation to reduce false positives"""
        text = text.strip()
        
        # Skip very short matches
        if len(text) < 2:
            return False
            
        # Skip common words that might match patterns
        common_words = {
            'english': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'],
            'hindi': ['और', 'या', 'में', 'पर', 'से', 'के', 'का', 'की', 'है', 'हैं', 'था', 'थी']
        }
        
        if text.lower() in common_words['english'] or text in common_words['hindi']:
            return False
            
        # Type-specific validation
        if pii_type == 'bank_account':
            # Must be reasonable length for bank account
            return 9 <= len(text) <= 18 and text.isdigit()
        elif pii_type == 'pincode':
            # Indian pincode validation
            return len(text) == 6 and text.isdigit() and not text.startswith('0')
        elif pii_type == 'aadhaar':
            # Aadhaar validation (basic)
            clean = re.sub(r'[\s-]', '', text)
            return len(clean) == 12 and clean.isdigit()
            
        return True
    
    def extract_all_pii(self, text: str) -> Dict[str, Any]:
        """Main function to extract all PII from text"""
        if not text or len(text.strip()) < 5:
            return {}
        
        # Extract using regex patterns
        regex_pii = self.extract_regex_pii(text)
        
        # Extract using NER models
        ner_pii = self.extract_ner_pii(text)
        
        # Combine and validate results
        final_pii = {}
        
        # Process regex-based PII
        for pii_type, matches in regex_pii.items():
            validated_matches = [m for m in matches if self.is_likely_pii(m, pii_type)]
            if validated_matches:
                final_pii[pii_type] = validated_matches
        
        # Process NER-based PII
        for category, entities in ner_pii.items():
            if entities:
                # Filter out very short entities and common words
                filtered_entities = [e for e in entities if len(e.strip()) > 2 and self.is_likely_pii(e, category)]
                if filtered_entities:
                    final_pii[category] = filtered_entities
        
        # Special validation for phones and emails
        if 'indian_mobile' in final_pii:
            final_pii['validated_phones'] = self.validate_phone_numbers(final_pii['indian_mobile'])
            
        if 'email' in final_pii:
            final_pii['validated_emails'] = self.validate_emails(final_pii['email'])
        
        return final_pii

class ElasticsearchConnector:
    def __init__(self, host: str, port: int = 9200, username: str = None, password: str = None, use_ssl: bool = False):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl
        self.es = None
        
    def connect(self) -> bool:
        """Connect to Elasticsearch"""
        try:
            if self.username and self.password:
                self.es = Elasticsearch(
                    [{'host': self.host, 'port': self.port}],
                    http_auth=(self.username, self.password),
                    use_ssl=self.use_ssl,
                    verify_certs=False,
                    timeout=30
                )
            else:
                self.es = Elasticsearch(
                    [{'host': self.host, 'port': self.port}],
                    use_ssl=self.use_ssl,
                    verify_certs=False,
                    timeout=30
                )
            
            # Test connection
            if self.es.ping():
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Connection failed: {e}")
            return False
    
    def get_indices(self) -> List[str]:
        """Get list of available indices"""
        try:
            if self.es:
                indices = self.es.indices.get_alias().keys()
                return sorted([idx for idx in indices if not idx.startswith('.')])
        except Exception as e:
            st.error(f"Error fetching indices: {e}")
        return []
    
    def search_data(self, index: str, query: Dict = None, size: int = 100) -> List[Dict]:
        """Search data from Elasticsearch"""
        try:
            if not query:
                query = {"match_all": {}}
            
            response = self.es.search(
                index=index,
                body={"query": query, "size": size}
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

def main():
    st.title("🔍 Multilingual PII Extraction Tool")
    st.markdown("Extract Personal Identifiable Information from Elasticsearch data (Hindi/English)")
    
    # Initialize session state
    if 'pii_extractor' not in st.session_state:
        st.session_state.pii_extractor = PIIExtractor()
        
    if 'es_connector' not in st.session_state:
        st.session_state.es_connector = None
        
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Elasticsearch configuration
        st.subheader("Elasticsearch Connection")
        es_host = st.text_input("Host", value="localhost", help="Elasticsearch host")
        es_port = st.number_input("Port", value=9200, help="Elasticsearch port")
        es_username = st.text_input("Username (optional)", help="Leave empty if no auth")
        es_password = st.text_input("Password (optional)", type="password")
        use_ssl = st.checkbox("Use SSL", help="Enable SSL connection")
        
        if st.button("Connect to Elasticsearch"):
            with st.spinner("Connecting..."):
                connector = ElasticsearchConnector(es_host, es_port, es_username, es_password, use_ssl)
                if connector.connect():
                    st.session_state.es_connector = connector
                    st.success("✅ Connected successfully!")
                else:
                    st.error("❌ Connection failed!")
        
        # Model loading
        st.subheader("Load NER Models")
        if st.button("Load Models"):
            with st.spinner("Loading models... This may take a few minutes"):
                st.session_state.pii_extractor.load_models()
                st.success("✅ Models loaded!")
    
    # Main interface
    if st.session_state.es_connector and st.session_state.es_connector.es:
        
        # Index selection
        st.subheader("📊 Select Data Source")
        indices = st.session_state.es_connector.get_indices()
        
        if indices:
            selected_index = st.selectbox("Choose Index", indices)
            
            # Query configuration
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query_type = st.selectbox(
                    "Query Type",
                    ["Match All", "Custom Query", "Text Search"]
                )
                
            with col2:
                max_docs = st.number_input("Max Documents", min_value=1, max_value=1000, value=100)
            
            # Build query
            query = {"match_all": {}}
            
            if query_type == "Text Search":
                search_text = st.text_input("Search Text")
                if search_text:
                    query = {
                        "multi_match": {
                            "query": search_text,
                            "fields": ["*"]
                        }
                    }
            elif query_type == "Custom Query":
                custom_query = st.text_area(
                    "Custom Elasticsearch Query (JSON)",
                    value='{"match_all": {}}',
                    height=100
                )
                try:
                    query = json.loads(custom_query)
                except json.JSONDecodeError:
                    st.error("Invalid JSON query")
                    query = {"match_all": {}}
            
            # Extract PII
            if st.button("🔍 Extract PII", type="primary"):
                with st.spinner("Extracting PII from Elasticsearch data..."):
                    
                    # Fetch data
                    documents = st.session_state.es_connector.search_data(selected_index, query, max_docs)
                    
                    if not documents:
                        st.warning("No documents found with the given query.")
                        return
                    
                    st.info(f"Processing {len(documents)} documents...")
                    
                    # Process documents
                    all_pii_results = []
                    progress_bar = st.progress(0)
                    
                    for idx, doc in enumerate(documents):
                        # Convert document to text
                        doc_text = ""
                        if isinstance(doc, dict):
                            # Extract text from all fields
                            for key, value in doc.items():
                                if isinstance(value, str):
                                    doc_text += f" {value}"
                                elif isinstance(value, (list, dict)):
                                    doc_text += f" {str(value)}"
                        else:
                            doc_text = str(doc)
                        
                        # Extract PII
                        pii_found = st.session_state.pii_extractor.extract_all_pii(doc_text)
                        
                        if pii_found:  # Only store documents with PII
                            all_pii_results.append({
                                'document_id': idx,
                                'document_preview': doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
                                'pii_found': pii_found,
                                'full_document': doc
                            })
                        
                        progress_bar.progress((idx + 1) / len(documents))
                    
                    st.session_state.extracted_data = all_pii_results
                    progress_bar.empty()
            
            # Display results
            if st.session_state.extracted_data:
                st.subheader("🎯 PII Extraction Results")
                
                total_docs_with_pii = len(st.session_state.extracted_data)
                st.metric("Documents with PII", total_docs_with_pii)
                
                # Summary statistics
                pii_summary = {}
                for result in st.session_state.extracted_data:
                    for pii_type, values in result['pii_found'].items():
                        if pii_type not in pii_summary:
                            pii_summary[pii_type] = 0
                        pii_summary[pii_type] += len(values)
                
                if pii_summary:
                    st.subheader("📈 PII Summary")
                    
                    # Create summary DataFrame
                    summary_df = pd.DataFrame([
                        {'PII Type': pii_type.replace('_', ' ').title(), 'Count': count}
                        for pii_type, count in pii_summary.items()
                    ]).sort_values('Count', ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.dataframe(summary_df, use_container_width=True)
                    
                    with col2:
                        st.bar_chart(summary_df.set_index('PII Type'))
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                
                # Filter options
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_pii_types = st.multiselect(
                        "Filter by PII Type",
                        options=list(pii_summary.keys()),
                        default=list(pii_summary.keys())
                    )
                
                with col2:
                    show_doc_preview = st.checkbox("Show Document Preview", value=True)
                
                # Display filtered results
                for i, result in enumerate(st.session_state.extracted_data):
                    # Filter based on selected PII types
                    filtered_pii = {k: v for k, v in result['pii_found'].items() if k in selected_pii_types}
                    
                    if not filtered_pii:
                        continue
                    
                    with st.expander(f"Document {result['document_id']} - {sum(len(v) for v in filtered_pii.values())} PII items found"):
                        
                        if show_doc_preview:
                            st.text_area("Document Preview", result['document_preview'], height=100, disabled=True)
                        
                        # Display PII by category
                        for pii_type, values in filtered_pii.items():
                            if values:
                                st.write(f"**{pii_type.replace('_', ' ').title()}:**")
                                
                                # Special handling for validated data
                                if pii_type == 'validated_phones':
                                    for phone_data in values:
                                        if phone_data['valid']:
                                            st.write(f"📱 {phone_data['formatted']} ({phone_data['country']}) - {phone_data['carrier']}")
                                        else:
                                            st.write(f"📱 {phone_data['original']} (validation failed)")
                                            
                                elif pii_type == 'validated_emails':
                                    for email_data in values:
                                        if email_data['valid']:
                                            st.write(f"📧 {email_data['normalized']}")
                                        else:
                                            st.write(f"📧 {email_data['original']} (validation failed)")
                                else:
                                    for value in values:
                                        st.write(f"• {value}")
                                
                                st.write("")  # Add spacing
                
                # Export functionality
                st.subheader("📤 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Export as JSON"):
                        json_data = json.dumps(st.session_state.extracted_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("Export as CSV"):
                        # Flatten data for CSV
                        csv_data = []
                        for result in st.session_state.extracted_data:
                            for pii_type, values in result['pii_found'].items():
                                for value in values:
                                    csv_data.append({
                                        'Document_ID': result['document_id'],
                                        'PII_Type': pii_type,
                                        'PII_Value': value,
                                        'Document_Preview': result['document_preview'][:100]
                                    })
                        
                        if csv_data:
                            df = pd.DataFrame(csv_data)
                            csv_string = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_string,
                                file_name=f"pii_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
                with col3:
                    if st.button("Clear Results"):
                        st.session_state.extracted_data = None
                        st.rerun()
        
        else:
            st.info("No indices found in Elasticsearch or connection not established.")
    
    else:
        st.info("👆 Please configure and connect to Elasticsearch in the sidebar to begin.")
    
    # Manual text input option
    st.subheader("✏️ Manual Text Analysis")
    st.markdown("Test PII extraction on custom text")
    
    manual_text = st.text_area(
        "Enter text for PII extraction",
        placeholder="Enter Hindi, English, or mixed language text here...",
        height=150
    )
    
    if st.button("Analyze Text") and manual_text:
        with st.spinner("Analyzing text..."):
            pii_results = st.session_state.pii_extractor.extract_all_pii(manual_text)
            
            if pii_results:
                st.subheader("Found PII:")
                for pii_type, values in pii_results.items():
                    st.write(f"**{pii_type.replace('_', ' ').title()}:** {', '.join(map(str, values))}")
            else:
                st.info("No PII found in the text.")

# Installation requirements info
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Required Packages")
st.sidebar.code("""
pip install streamlit
pip install elasticsearch
pip install spacy
pip install transformers
pip install phonenumbers
pip install email-validator
pip install pandas
python -m spacy download en_core_web_sm
""")

if __name__ == "__main__":
    main()
