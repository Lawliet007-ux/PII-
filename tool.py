import streamlit as st
import re
from elasticsearch import Elasticsearch
from transformers import pipeline

# Load the NER model for Hindi-English code-mixed text
ner_pipeline = pipeline('ner', model="sagorsarker/codeswitch-hineng-ner-lince", aggregation_strategy="simple")

# Function to extract PII using NER and regex
def extract_pii(text):
    # NER extraction
    ner_results = ner_pipeline(text)
    
    pii = {
        'PERSON': [],
        'LOCATION': [],
        'ORGANIZATION': [],
        'TIME': [],
        'DATE': [],  # Will use regex for dates
        'FIR_NUMBER': [],
        'OTHER': []
    }
    
    # Process NER results
    for entity in ner_results:
        if entity['score'] > 0.7:  # Threshold for accuracy, adjust if needed for balance between recall and precision
            entity_type = entity['entity_group']
            if entity_type in ['PER', 'PERSON']:
                pii['PERSON'].append(entity['word'])
            elif entity_type in ['LOC', 'LOCATION']:
                pii['LOCATION'].append(entity['word'])
            elif entity_type in ['ORG', 'ORGANIZATION']:
                pii['ORGANIZATION'].append(entity['word'])
            elif entity_type == 'TIME':
                pii['TIME'].append(entity['word'])
            else:
                pii['OTHER'].append(f"{entity_type}: {entity['word']}")
    
    # Regex for dates (e.g., 19/11/2017)
    dates = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
    pii['DATE'].extend(dates)
    
    # Regex for times (e.g., 21:33)
    times = re.findall(r'\b\d{2}:\d{2}\b', text)
    pii['TIME'].extend(times)
    
    # Regex for FIR numbers (e.g., FIR No. 0523)
    fir_numbers = re.findall(r'FIR No\.?\s*[:.]?\s*(\d+)', text, re.IGNORECASE)
    pii['FIR_NUMBER'].extend(fir_numbers)
    
    # Remove duplicates
    for key in pii:
        pii[key] = list(set(pii[key]))
    
    return pii

# Streamlit UI
st.title("PII Extractor from Elasticsearch Data")

# Input fields for Elasticsearch connection and query
es_host = st.text_input("Elasticsearch Host", value="localhost")
es_port = st.number_input("Elasticsearch Port", value=9200)
es_index = st.text_input("Elasticsearch Index")
text_field = st.text_input("Text Field in Documents", value="content")
query_json = st.text_area("Elasticsearch Query (JSON)", value='{"query": {"match_all": {}}}')

if st.button("Extract PII"):
    try:
        # Connect to Elasticsearch
        es = Elasticsearch([f"http://{es_host}:{es_port}"])
        
        # Execute search
        res = es.search(index=es_index, body=query_json)
        
        hits = res['hits']['hits']
        if not hits:
            st.write("No documents found.")
        else:
            for i, hit in enumerate(hits):
                doc_id = hit['_id']
                text = hit['_source'].get(text_field, "")
                
                if text:
                    st.subheader(f"Document ID: {doc_id}")
                    pii = extract_pii(text)
                    
                    # Display extracted PII
                    for category, items in pii.items():
                        if items:
                            st.write(f"**{category}:** {', '.join(items)}")
                else:
                    st.write(f"No text found in field '{text_field}' for document {doc_id}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Additional notes
st.info("""
This tool extracts PII such as names (PERSON), locations (LOCATION), organizations (ORGANIZATION), times (TIME), dates (DATE), and FIR numbers from Elasticsearch documents.
It uses a Hugging Face model specialized for Hindi-English code-mixed NER and regex for additional patterns.
Ensure the model and libraries are installed: pip install streamlit elasticsearch transformers torch.
Run the app with: streamlit run app.py
For better handling of OCR errors, consider preprocessing the text (e.g., removing special characters), but this is not implemented here to avoid altering potential PII.
""")
