# logic.py
# this whole file was made by chatgpt with the prompt perfect this file then i put the file i had
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np

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
    ID_TO_LABEL = {1: 'IMMORAL', 0: 'MORAL'} 
    
except Exception as e:
    print(f"FATAL ERROR: Could not load moral classifier from {MORAL_MODEL_PATH}.")
    print(f"Please ensure the model is saved correctly. Error: {e}")
    moral_model = None
    


def process_input(user_text):
    # Replace this with your custom logic
    return f"You typed: {user_text.upper()}"


# --- CLASSIFICATION FUNCTION ---
def predict_moral_status(text):
    """Classifies a single text input as 0 (IMMORAL) or 1 (MORAL)."""
    if moral_model is None:
        return np.nan # Cannot proceed if model is missing
        
    # Use the moral classifier's tokenizer and model
    enc = moral_tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding='max_length', # Consistent padding for batching (though only single item here)
        max_length=128
    ).to(DEVICE)
    print(f"DEBUG: Input tokens: {enc}")
    with torch.no_grad():
        out = moral_model(**enc)
    print(f"DEBUG: Model output logits: {out.logits}")
    # Get the predicted class ID (0 or 1)
    pred_id = torch.argmax(out.logits, dim=-1).item()
    if pred_id:
        print(f"MORAL! We think this action is morally acceptable. In this situation {sit}. It has this intention {intent}. Which does is good because this norm is good {norm}.")
    else:
        print(f"IMMORAL! We think this action is morally unacceptable. Because of this norm {norm}. In this situation {sit}. It has this intention {intent}. Which does is bad.")
    return pred_id