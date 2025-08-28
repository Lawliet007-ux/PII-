# tool.py
import streamlit as st
from transformers import pipeline
import re
import json

# -------------------------
# Load HuggingFace Multilingual Model
# -------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "token-classification",
        model="Davlan/xlm-roberta-base-ner-hrl",  # multilingual NER
        aggregation_strategy="simple"
    )

ner_pipeline = load_model()

# -------------------------
# Regex Extractor for Legal PII
# -------------------------
def regex_extract(text: str):
    pii = []

    # FIR No
    for f in re.findall(r"FIR\s*No.*?(\d+)", text, flags=re.IGNORECASE):
        pii.append({"label": "FIR_NO", "text": f})

    # Dates
    for d in re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text):
        pii.append({"label": "DATE", "text": d})

    # Times
    for t in re.findall(r"\d{1,2}:\d{2}", text):
        pii.append({"label": "TIME", "text": t})

    # Year
    for y in re.findall(r"\b(19|20)\d{2}\b", text):
        pii.append({"label": "YEAR", "text": y})

    # Sections of Law
    for s in re.findall(r"(?:Section|Sec)\s*\d+[A-Za-z\-]*", text, flags=re.IGNORECASE):
        pii.append({"label": "SECTION", "text": s})

    # GD / Diary No
    for g in re.findall(r"Diary\s*Reference.*?(\d+)", text, flags=re.IGNORECASE):
        pii.append({"label": "GD_REFERENCE", "text": g})

    # Address
    for a in re.findall(r"Address[:\- ]+([^\n]+)", text, flags=re.IGNORECASE):
        pii.append({"label": "ADDRESS", "text": a.strip()})

    # Police Station
    for ps in re.findall(r"P\.?S\.?\s*[:\- ]+([^\n]+)", text, flags=re.IGNORECASE):
        pii.append({"label": "POLICE_STATION", "text": ps.strip()})

    return pii

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Multilingual PII Extractor", layout="wide")
st.title("🔍 Multilingual Legal PII Extractor")

sample_text = """P.S. (Police Thane): Bhosari
FIR No.: 0523
Date and Time of FIR: 19/11/2017 at 21:33
District: Pune City
Year: 2017
Acts / Sections: Section 25, Section 135
General Diary Reference: 029
Date To: 19/11/2017
Time From: 17:15 hours
Address: Shoa Hate Sadma, Moya Mat, Ashita, Asaravadi, Pune"""

user_text = st.text_area("Paste Legal FIR / Case Text", value=sample_text, height=300)

if st.button("Extract PII"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        # Run multilingual NER
        ner_results = ner_pipeline(user_text)

        ner_extracted = []
        for r in ner_results:
            ner_extracted.append({
                "label": r["entity_group"],
                "text": r["word"],
                "score": float(r["score"])  # ensure JSON-safe
            })

        # Regex PII
        regex_results = regex_extract(user_text)

        # Merge
        final_pii = ner_extracted + regex_results

        # Display
        st.subheader("📌 Extracted PII")
        if final_pii:
            for item in final_pii:
                if "score" in item:
                    st.write(f"**{item['label']}** → {item['text']} (score={item['score']:.2f})")
                else:
                    st.write(f"**{item['label']}** → {item['text']}")
        else:
            st.info("No PII detected.")

        # JSON-safe export
        safe_json = json.dumps(final_pii, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download JSON",
            data=safe_json,
            file_name="pii_extracted.json",
            mime="application/json"
        )
