import streamlit as st
import spacy
from PyPDF2 import PdfReader
import tempfile
import os
import pickle
import random

# Load training data
train_data = pickle.load(open('D:/NEW RESUME GIT/Resume_Parser_Test/Train.pkl', 'rb'))
print (train_data)
# Define function to train the model
def train_model(train_data):
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        for ent in annotations['entities']:
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                try:
                    doc = nlp.make_doc(text)
                    example = spacy.training.Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.2, losses=losses)
                except Exception as e:
                    pass

    return nlp

# Train the model (comment out if the model is already trained and saved)
trained_nlp = train_model(train_data)

# Save the trained model (comment out if the model is already trained and saved)
trained_nlp.to_disk('nlp_model')

# Load the pre-trained NLP model
nlp_model = spacy.load('nlp_model')

def process_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf = PdfReader(f)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        doc = nlp_model(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
        validated_entities = [(label, value) for label, values in entities.items() for value in values]
        return validated_entities

# Streamlit app
st.title("PDF NER Extraction")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Process the uploaded file
    entities = process_pdf(temp_file_path)

    # Remove the temporary file
    os.remove(temp_file_path)

    if entities:
        st.write("NER predictions:")
        for label, text in entities:
            st.write(f"{label.upper():{30}} - {text}")
    else:
        st.write("No entities found in the PDF.")
