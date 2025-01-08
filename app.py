import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/medical_summarization")

# Streamlit app title
st.title("Medical Note Summarization")

# Input text area for medical note
medical_note = st.text_area("Enter the medical note here:", height=400, placeholder="insert the medical note here...")

# Button to trigger summarization
if st.button("Summarize"):
    if medical_note:
        try:
            with st.spinner("Generating summary..."):
                inputs = tokenizer.encode("summarize: " + medical_note, return_tensors="pt", max_length=512, truncation=True)
                summary_ids = model.generate(inputs, max_length=250, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.subheader("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a medical note to summarize.")