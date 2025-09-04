import streamlit as st
import pdfplumber
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering
import spacy
from spacy import displacy
from collections import defaultdict
import re
import torch
from functools import lru_cache

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

# Load models with caching
@st.cache_resource
def load_ner_model():
    """Load a multilingual NER model"""
    try:
        # Using a model that supports multiple languages including Hindi
        return pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="average")
    except:
        # Fallback to a more common model
        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

@st.cache_resource
def load_qa_model():
    """Load a question answering model"""
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
        # If model isn't downloaded, try to download it
        try:
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except:
            return None

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using pdfplumber with improved handling"""
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None
    return text

def extract_pii_with_ner(text, ner_pipeline):
    """Extract PII using NER model"""
    if not text or not ner_pipeline:
        return {}
    
    # Process text in chunks if it's too long
    max_chunk_size = 512
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    entities = []
    for chunk in chunks:
        try:
            result = ner_pipeline(chunk)
            entities.extend(result)
        except Exception as e:
            st.warning(f"Error processing chunk: {str(e)}")
            continue
    
    # Organize entities by type
    pii_data = defaultdict(list)
    for entity in entities:
        entity_type = entity['entity_group']
        entity_text = entity['word']
        
        # Clean up the entity text
        if entity_text.startswith('##'):
            continue
            
        pii_data[entity_type].append(entity_text)
    
    # Remove duplicates while preserving order
    for key in pii_data:
        pii_data[key] = list(dict.fromkeys(pii_data[key]))
    
    return dict(pii_data)

def extract_structured_info(qa_pipeline, text, questions):
    """Extract structured information using question answering"""
    results = {}
    if not qa_pipeline:
        return results
    
    for field, question in questions.items():
        try:
            answer = qa_pipeline(question=question, context=text)
            if answer['score'] > 0.1:  # Threshold for confidence
                results[field] = answer['answer']
            else:
                results[field] = "Not found"
        except:
            results[field] = "Error extracting"
    
    return results

def visualize_entities(text, ner_results):
    """Visualize named entities in text"""
    if not text:
        return
    
    # Convert to spaCy format for visualization
    colors = {
        "PER": "#F8B195", 
        "ORG": "#F67280", 
        "LOC": "#C06C84", 
        "MISC": "#6C5B7B"
    }
    
    options = {"colors": colors}
    
    # Create a simple visualization
    doc = {"text": text, "ents": [], "title": None}
    
    for entity_type, entities in ner_results.items():
        for entity in entities:
            # Find all occurrences of this entity in the text
            for match in re.finditer(re.escape(entity), text):
                doc["ents"].append({
                    "start": match.start(),
                    "end": match.end(),
                    "label": entity_type
                })
    
    # Display using spaCy's visualizer
    if doc["ents"]:
        html = displacy.render(doc, style="ent", options=options, manual=True, page=True)
        st.components.v1.html(html, height=300, scrolling=True)

def main():
    st.title("📄 FIR PII Extraction Tool")
    st.markdown("Extract Personal Identifiable Information from FIR documents in multiple Indian languages")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload FIR Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        st.header("Settings")
        extraction_method = st.radio(
            "Extraction Method",
            ["NER Only", "QA Only", "Combined Approach"],
            help="Choose the method for information extraction"
        )
        
        st.header("About")
        st.info("This tool uses advanced NLP techniques to extract PII from FIR documents in multiple Indian languages.")
    
    # Main content area
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(uploaded_file)
            
            if extracted_text:
                st.session_state.extracted_text = extracted_text
                
                # Display extracted text
                with st.expander("View Extracted Text"):
                    st.text_area("Text", extracted_text, height=300)
                
                # Load models
                with st.spinner("Loading NLP models..."):
                    ner_pipeline = load_ner_model()
                    qa_pipeline = load_qa_model()
                    nlp = load_spacy_model()
                
                # Extract information based on selected method
                if extraction_method in ["NER Only", "Combined Approach"]:
                    with st.spinner("Extracting PII with NER..."):
                        ner_results = extract_pii_with_ner(extracted_text, ner_pipeline)
                        st.session_state.pii_data.update(ner_results)
                
                if extraction_method in ["QA Only", "Combined Approach"] and qa_pipeline:
                    with st.spinner("Extracting structured information with QA..."):
                        # Define questions for different fields
                        questions = {
                            "fir_no": "What is the FIR number?",
                            "year": "What is the year of the FIR?",
                            "state_name": "Which state is mentioned in the FIR?",
                            "dist_name": "Which district is mentioned in the FIR?",
                            "police_station": "Which police station is mentioned in the FIR?",
                            "under_acts": "Under which acts is the case registered?",
                            "under_sections": "Which sections are mentioned in the FIR?",
                            "revised_case_category": "What is the case category?",
                            "oparty": "Who is the opposite party or accused?"
                        }
                        
                        qa_results = extract_structured_info(qa_pipeline, extracted_text, questions)
                        st.session_state.pii_data.update(qa_results)
                
                # Display results
                st.subheader("Extracted Information")
                
                if st.session_state.pii_data:
                    # Create two columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Personal Identifiable Information**")
                        pii_df = pd.DataFrame.from_dict(
                            {k: [", ".join(v)] if isinstance(v, list) else [v] 
                             for k, v in st.session_state.pii_data.items() if k in ["PER", "LOC", "ORG", "MISC"]}, 
                            orient='index', 
                            columns=['Values']
                        )
                        st.dataframe(pii_df)
                    
                    with col2:
                        st.write("**FIR Details**")
                        fir_details = {k: v for k, v in st.session_state.pii_data.items() 
                                     if k not in ["PER", "LOC", "ORG", "MISC"]}
                        fir_df = pd.DataFrame.from_dict(
                            {k: [v] for k, v in fir_details.items()}, 
                            orient='index', 
                            columns=['Values']
                        )
                        st.dataframe(fir_df)
                    
                    # Visualization
                    st.subheader("Entity Visualization")
                    visualize_entities(extracted_text, 
                                      {k: v for k, v in st.session_state.pii_data.items() 
                                       if k in ["PER", "LOC", "ORG", "MISC"]})
                    
                    # Download button
                    csv = pd.DataFrame.from_dict(st.session_state.pii_data, orient='index').to_csv().encode('utf-8')
                    st.download_button(
                        label="Download extracted data as CSV",
                        data=csv,
                        file_name="extracted_pii.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No PII could be extracted from the document.")
            else:
                st.error("Could not extract text from the uploaded PDF.")
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
            - Names (Complainant/Accused)
            - Addresses and Locations
            - FIR Number and Details
            - Legal Sections and Acts
            - Case Categories
            """)
        
        with col3:
            st.subheader("Technology")
            st.markdown("""
            - Transformer-based NER
            - Question Answering Models
            - Advanced PDF text extraction
            - Multilingual support
            """)

if __name__ == "__main__":
    main()
