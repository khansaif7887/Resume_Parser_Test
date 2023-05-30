import spacy
import random
import streamlit as st
from spacy.training.example import Example
from PyPDF2 import PdfReader
import pickle

# Load the training data from a .pkl file
with open("data/training/train_data.pkl", "rb") as f:
    training_data = pickle.load(f)

# Initialize a blank spaCy model
nlp = spacy.blank('en')

# Define the entity labels you want to recognize
entity_labels = ['Companies worked at', 'Skills']

# Create the entity recognizer and add it to the pipeline
ner = nlp.create_pipe('ner')
nlp.add_pipe('ner')

# Add labels to the entity recognizer
for label in entity_labels:
    ner.add_label(label)

# Process and correct the training data

processed_data = []
try:
    for text, annotations in training_data:
        corrected_spans = []
        for start, end, label in annotations.get('entities', []):
            if label in entity_labels:
                overlap = any(start < span_end and span_start < end for span_start, span_end, _ in corrected_spans)
                if not overlap:
                    corrected_spans.append((start, end, label))
        corrected_annotations = {'entities': corrected_spans}
        processed_data.append((text, corrected_annotations))
except KeyError:
    # Handle the KeyError here
    # For example, you can print an error message or perform any desired action
    print("KeyError: Transition 'O' not found. Skipping...")

# Update the model with the corrected training data
for text, annotations in processed_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    spans = []
    try:
        for start, end, label in annotations.get('entities', []):
            overlapping = any(start < span_end and span_start < end for span_start, span_end, _ in spans)
            if not overlapping:
                spans.append((start, end, label))
        with doc.retokenize() as retokenizer:
            for span_start, span_end, label in spans:
                retokenizer.merge(doc[span_start:span_end], attrs={"label": label})
        example = Example.from_dict(doc, annotations)
        nlp.update([example])
    except KeyError:
        # Handle the KeyError here
        # For example, you can print an error message or perform any desired action
        print("KeyError: Transition 'O' not found. Skipping...")
    except ValueError:
        # Handle the ValueError here
        # For example, you can print an error message or perform any desired action
        print("ValueError: Conflicting entities found. Skipping...")
    except Exception as e:
        # Handle other specific exceptions or use a more generic exception type
        # For example, print the error message or perform any desired action
        print(f"An error occurred during nlp.update(): {e}")



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
nlp.to_disk('nlp_model')

# Load the trained model
nlp_model = spacy.load('nlp_model')

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