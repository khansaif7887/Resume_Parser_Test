import spacy
import pickle

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Read the sample CV text
with open("D:/NEW RESUME GIT/Resume_Parser_Test/Resume/data/test/John.txt", "r") as f:
    cv_text = f.read()

# Process the CV text with the NER model
doc = nlp(cv_text)

# Define the entity types to extract and their corresponding labels
entity_types = {
    "NAME": "PERSON",
    "MOBILE": "PHONE",
    "SKILLS": "SKILLS",
    "EXPERIENCE": "EXPERIENCE",
    "ADDRESS": "GPE",
    "EMAIL": "EMAIL"
}

# Create the training data in spaCy format
train_data = []
for ent in doc.ents:
    if ent.label_ in entity_types.values():
        label = [key for key, value in entity_types.items() if value == ent.label_][0]
        train_data.append((ent.text, {"entities": [(ent.start_char, ent.end_char, label)]}))

# Save the training data as a .pkl file
with open("training_data.pkl", "wb") as f:
    pickle.dump(train_data, f)
