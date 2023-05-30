import spacy
import random
import streamlit as st
from spacy.training.example import Example
from PyPDF2 import PdfReader

# Load the converted training data
training_data_file = "data/training/train.spacy"
nlp = spacy.blank('en')
nlp.initialize()

# Load the converted training data
training_data = list(spacy.util.load_data(training_data_file))

# Define the entity labels you want to recognize
entity_labels = ['Companies worked at', 'Skills']

# Create the entity recognizer and add it to the pipeline
ner = nlp.add_pipe("ner")

# Add labels to the entity recognizer
for label in entity_labels:
    ner.add_label(label)

# Disable other pipeline components during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for iteration in range(10):
        random.shuffle(training_data)
        losses = {}
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)

# Save the trained model
output_dir = 'nlp_model'
nlp.to_disk(output_dir)

# Load the trained model
nlp_model = spacy.load(output_dir)

# Function to extract named entities from a PDF file
def extract_named_entities_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit app code
st.title("PDF NER Extraction")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract named entities from the uploaded PDF
    pdf_entities = extract_named_entities_from_pdf(uploaded_file)

    # Print the extracted named entities
    st.header("Extracted Named Entities")
    for entity, label in pdf_entities:
        st.write(f"{label}: {entity}")
