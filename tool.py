import json
st.title("🔎 Multilingual PII Extractor for Legal OCR (EN + Indic)")
st.caption("Hybrid rules + Indic NER. Designed to minimize false positives and misses.")


with st.sidebar:
    st.header("Model & Settings")
    model_name = st.text_input(
        "HF model",
        value="ai4bharat/IndicNER",
        help="Change to your fine-tuned model if available."
    )
    threshold = st.slider(
        "NER confidence threshold",
        0.0,
        1.0,
        0.55,
        0.01,
        help="Higher = fewer false positives, but may miss low-confidence entities in noisy OCR."
    )
    apply_regex = st.checkbox("Apply regex rules", value=True)
    apply_ner = st.checkbox("Apply NER model", value=True)
    mask_token = st.text_input("Mask token for redaction", value="▮▮▮")

    st.markdown("---")
    st.write("**Tips**: For very noisy OCR, lower threshold to ~0.45 and rely more on regex for dates/IDs.")

nlp = load_ner_pipeline(model_name) if st.session_state.get("loaded_model") != model_name else st.session_state.get("nlp")
st.session_state["loaded_model"] = model_name
st.session_state["nlp"] = nlp


sample_text = (
"1.\nP.S. (Police Thane): Bhosari \nFIR No. (C.R.): 0523 \nDate and Time of FIR: \n19/11/2017 at 21:33 \nDistrict: Pune City \nYear: 2017\n"
"2.\nS.No. \nActs \nSections \n1 \n \n25 \n 3\n3 \n Maharashtra Police \n135\n (a) \nOccurrence of offence: \n (b) \nInformation received at P.S.: \nDate: 19/11/2017 \nTime: 21:09 hours \n (c) \nGeneral Diary Reference: \n029\nDay: Sunday \nDate from: 19/11/2017 \nDate To: 19/11/2017\nTime Period: \nTime From: 17:15 hours \n (a) Direction and distance from P.S.: N, 2 Min. \n (b) Address: Shoba Hate Sadma, Maya Mata, Ashit, Asarawadi, Pune\n"
)


text = st.text_area(
"Paste Elasticsearch text here (any language/mixed)",
value=sample_text,
height=260,
)


col_run, col_clear = st.columns([1, 1])
with col_run:
run = st.button("Extract PII", type="primary")
with col_clear:
if st.button("Clear"):
st.experimental_rerun()


if run:
raw = text
norm = normalize_ocr(raw)


spans: List[Span] = []
if apply_regex:
spans.extend(run_regex_rules(norm))
if apply_ner and nlp is not None:
spans.extend(run_ner(norm, nlp, threshold))


spans = filter_false_positives(spans)
spans = dedupe_spans(spans)


grouped = group_by_label(spans)


st.subheader("Results")
c1, c2 = st.columns(2)
with c1:
st.markdown("**Normalized Input**")
st.code(norm, language="text")
st.markdown("**Redaction Preview**")
st.code(redact_text(norm, spans, mask_token=mask_token), language="text")


with c2:
st.markdown("**Extracted PII (JSON)**")
st.json(grouped)


# Download JSON
st.download_button(
label="Download PII JSON",
data=json.dumps(grouped, ensure_ascii=False, indent=2),
file_name="pii_extracted.json",
mime="application/json",
)


# Simple metrics
st.markdown("---")
st.write(f"**Total entities:** {len(spans)}")
label_counts = {k: len(v) for k, v in grouped.items()}
if label_counts:
st.write("**Counts by type:**", label_counts)


st.markdown("---")
st.markdown(
"**Notes**: \n"
"- Swap in your fine-tuned HF NER for best accuracy on FIRs. \n"
"- Extend regexes for other IDs (e.g., VoterID, DL, Passport) as needed. \n"
"- To avoid over-redaction, keep the threshold moderate and use the blacklist in `filter_false_positives()`.\n"
)
