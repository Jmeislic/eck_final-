# logic.py
# this whole file was made by chatgpt with the prompt I am confused I have a seperate file for the front end in html how do i combine it with this also how do I get this to work I want the screen to request an input then i take the input run it through a python function in a seperate file then get an answer and display that on the screen from flask import Flask, render_template app = Flask(__name__) # Define a route for the root URL ('/') @app.route('/') def index(): # Fetch data from the database and prepare for rendering data = get_data_from_database() # Replace this with your actual data retrieval logic # Render the 'index.html' template and pass the retrieved data for rendering return render_template('index.html', data=data) # Placeholder for fetching data from the database def get_data_from_database(): # Replace this function with your actual logic to retrieve data from the database # For now, returning a sample data return {'message': 'Hello, data from the database!'} if __name__ == '__main__': # Run the Flask application app.run(debug=True) <!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8"> <meta name="viewport" content="width=device-width, initial-scale=1.0"> <title>Flask App</title> </head> <body> <h1>Data from the Database</h1> <p>{{ data ss }}</p> <!-- Use the 'data' variable in the template --> </body> </html>
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np

MAX_LEN = 128
# --- CONFIGURATION (Based on your provided script) ---
MORAL_MODEL_PATH = "./bert_moral_classifier"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FILE PATHS ---
# Replace 'test_data.csv' with the actual path to your CSV file
TEST_CSV_PATH = 'testingSets/dataSet_1_train.csv'

# --- LOAD MORAL CLASSIFIER ---
try:
    print(f"Loading moral classifier from {MORAL_MODEL_PATH}...")
    moral_tokenizer = BertTokenizerFast.from_pretrained(MORAL_MODEL_PATH)
    moral_model = BertForSequenceClassification.from_pretrained(MORAL_MODEL_PATH)
    moral_model.to(DEVICE)
    moral_model.eval()
    
    # Map model output to labels (assuming 0 is 'IMMORAL' and 1 is 'MORAL' from your training setup)
    ID_TO_LABEL = {0: 'IMMORAL', 1: 'MORAL'} 
    
except Exception as e:
    print(f"FATAL ERROR: Could not load moral classifier from {MORAL_MODEL_PATH}.")
    print(f"Please ensure the model is saved correctly. Error: {e}")
    moral_model = None
    


def process_input(user_text):
    # Replace this with your custom logic
    return f"You typed: {user_text.upper()}"


labels = ["O", "B-SITUATION", "I-SITUATION", "B-NORM", "I-NORM", "B-INTENTION", "I-INTENTION", "B-ACTION", "I-ACTION"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

def get_spans(tokens, predictions):
    """Converts token predictions back into human-readable component spans."""
    spans = {"SITUATION": "", "NORM": "", "INTENTION": "", "ACTION": ""}
    current_label = None
    span_tokens = []
    
    for token, pred_id in zip(tokens, predictions):
        label = id2label[pred_id]
        
        # Handle end of a span (O or a new B-tag)
        if label == "O" or label.startswith("B-"):
            if current_label and span_tokens:
                # Append the completed span
                spans[current_label] += " " + moral_tokenizer.convert_tokens_to_string(span_tokens)
            
            # Reset for O-tag or start of new span
            if label == "O":
                current_label = None
                span_tokens = []
            elif label.startswith("B-"):
                current_label = label.split("-")[1]
                span_tokens = [token]
        
        # Handle I-tag continuance
        elif label.startswith("I-") and current_label == label.split("-")[1]:
            span_tokens.append(token)
            
        # Handle unexpected tag (I-tag without preceding B-tag or a label switch)
        else:
            if current_label and span_tokens:
                spans[current_label] += " " + moral_tokenizer.convert_tokens_to_string(span_tokens)
            current_label = None
            span_tokens = []
            
    # Capture the last remaining span
    if current_label and span_tokens:
        spans[current_label] += " " + moral_tokenizer.convert_tokens_to_string(span_tokens)
        
    for k in spans:
        spans[k] = spans[k].strip()
        
    return spans


def predict_moral_status(text):
    """Performs component extraction and moral classification."""
    if moral_model is None:
        return {"SITUATION": "Model Error", "NORM": "Model Error", "INTENTION": "Model Error", "ACTION": "Model Error"}, "UNKNOWN (Classifier Missing)"

    # Extract components
    
    enc = moral_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    with torch.no_grad():
        outputs = moral_model(**enc)
    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    tokens = moral_tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    spans = get_spans(tokens, preds)
    
    # Moral classification
    combined_text = " ".join([spans[k] for k in ['SITUATION','NORM','INTENTION','ACTION'] if spans[k]])
    if combined_text.strip() == "":
        combined_text = text  # fallback
    
    enc2 = moral_tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out2 = moral_model(**enc2)
    pred_label = torch.argmax(out2.logits, dim=-1).item()
    moral_status = "MORAL" if pred_label == 1 else "IMMORAL"
    for k, v in spans.items():
        print(f"{k}: {v}")
    # if moral_status == "MORAL":
    #     print(f"MORAL! We think this action is morally acceptable. In this situation {sit}. It has this intention {intent}. Which does is good because this norm is good {norm}.")
    # else:
    #     print(f"IMMORAL! We think this action is morally unacceptable. Because of this norm {norm}. In this situation {sit}. It has this intention {intent}. Which does is bad.")
 

    return spans, moral_status