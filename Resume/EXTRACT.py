import PyPDF2
import spacy
import pickle
# Open the PDF file in binary mode
with open("Resume/data/test/John Smith.pdf", "rb") as f:
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(f)
    # Extract the text from the PDF file
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Convert the PDF text to a spaCy Doc object
doc = nlp(pdf_text)# Define the entities to extract
entities = ["PERSON", "ORG", "GPE", "PHONE", "EMAIL"]

# Extract the entities from the Doc object
data = []
for ent in doc.ents:
    if ent.label_ in entities:
        data.append((ent.text, ent.label_))

# Save the extracted data as a training data file
with open("Train.pkl", "wb") as f:
    pickle.dump(data, f)
