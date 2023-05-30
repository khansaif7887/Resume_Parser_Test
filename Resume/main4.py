import streamlit as st
import spacy
from PyPDF2 import PdfReader
import tempfile
import os
import pickle
import random
import textract
import docx2txt

# Load training data
train_data = pickle.load(open('D:/NEW RESUME GIT/Resume_Parser_Test/training_data.pkl', 'rb'))

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

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        doc = nlp_model(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
        validated_entities = [(label, value) for label, values in entities.items() for value in values]
        return validated_entities

def process_doc(file_path):
    text = docx2txt.process(file_path)
    doc = nlp_model(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = set()
        entities[ent.label_].add(ent.text)
    validated_entities = [(label, value) for label, values in entities.items() for value in values]
    return validated_entities

# Streamlit app
st.title("NER Extraction")
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Process the uploaded file based on the file type
    file_ext = os.path.splitext(temp_file_path)[1]
    if file_ext == ".pdf":
        entities = process_pdf(temp_file_path)
    elif file_ext == ".txt":
        entities = process_text(temp_file_path)
    elif file_ext == ".docx":
        entities = process_doc(temp_file_path)
    else:
        entities = []

    # Remove the temporary file
    os.remove(temp_file_path)

    if entities:
        st.write("NER predictions:")
        for label, text in entities:
            st.write(f"{label.upper():{30}} - {text}")
    else:
        st.write("No entities found in the file.")
