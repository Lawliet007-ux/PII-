# tool.py
import streamlit as st
from transformers import pipeline
import re

# -------------------------
# Load HuggingFace NER Model
# -------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

ner_pipeline = load_model()

# -------------------------
# Regex-based Extractor
# -------------------------
def regex_extract(text: str):
    pii = []

    # FIR No
    fir = re.findall(r"FIR\s*No.*?(\d+)", text, flags=re.IGNORECASE)
    for f in fir:
        pii.append({"label": "FIR_NO", "text": f})

    # Dates (dd/mm/yyyy)
    dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text)
    for d in dates:
        pii.append({"label": "DATE", "text": d})

    # Times (hh:mm)
    times = re.findall(r"\d{1,2}:\d{2}", text)
    for t in times:
        pii.append({"label": "TIME", "text": t})

    # Year
    year = re.findall(r"Year.*?(\d{4})", text, flags=re.IGNORECASE)
    for y in year:
        pii.append({"label": "YEAR", "text": y})

    # Sections of IPC/Acts
    sections = re.findall(r"Section[s]?\s*\d+[A-Za-z\-]*", text, flags=re.IGNORECASE)
    for s in sections:
        pii.append({"label": "SECTION", "text": s})

    # Address (roughly after keyword 'Address' or 'P.S.')
    addresses = re.findall(r"(Address.*?[,.\n].*?)(?:\n|$)", text, flags=re.IGNORECASE)
    for a in addresses:
        pii.append({"label": "ADDRESS", "text": a.strip()})

    return pii

# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="PII Extractor", layout="wide")
st.title("🔍 Multilingual PII Extractor (Legal Texts)")

sample_text = """1.\nP.S. (Police Thane): Bhosari \nFIR No.: 0523 \nDate and Time of FIR: \n19/11/2017 at 21:33 \nDistrict: Pune City \nYear: 2017
Acts \nSections \n1 \n Section 25 \n3 \n Section 135
Occurrence of offence: \nDate: 19/11/2017 \nTime: 21:09 hours
General Diary Reference: 029
Date To: 19/11/2017
Time From: 17:15 hours
Address: Shoa Hate Íya Sadma, Moya Mat, Ashita, Asaravadi, Pune"""

with st.container():
    st.subheader("Enter / Paste Legal Text")
    user_text = st.text_area("Text Input", value=sample_text, height=300)

if st.button("Extract PII"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        # Run HuggingFace NER
        ner_results = ner_pipeline(user_text)

        # Convert to simpler dict
        ner_extracted = []
        for r in ner_results:
            ner_extracted.append({
                "label": r["entity_group"],
                "text": r["word"],
                "score": round(r["score"], 3)
            })

        # Run Regex Extractor
        regex_results = regex_extract(user_text)

        # Merge both results
        final_pii = ner_extracted + regex_results

        # Display results
        st.subheader("📌 Extracted PII")
        if final_pii:
            for item in final_pii:
                if "score" in item:
                    st.write(f"**{item['label']}** → {item['text']} (score: {item['score']})")
                else:
                    st.write(f"**{item['label']}** → {item['text']}")
        else:
            st.info("No PII detected.")

        # Download as JSON
        import json
        st.download_button(
            label="📥 Download PII (JSON)",
            data=json.dumps(final_pii, indent=2, ensure_ascii=False),
            file_name="pii_extracted.json",
            mime="application/json"
        )
