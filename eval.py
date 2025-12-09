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
    
    with torch.no_grad():
        out = moral_model(**enc)
    
    # Get the predicted class ID (0 or 1)
    pred_id = torch.argmax(out.logits, dim=-1).item()
    return pred_id


# --- MAIN EVALUATION LOGIC ---
if moral_model is not None:
    print(f"Reading test data from {TEST_CSV_PATH}...")
    try:
        # Load the CSV. Assumes 'label' is the true value (0 or 1) and 'input' is the text.
        df = pd.read_csv(TEST_CSV_PATH)
        
        # --- PREPARE DATA ---
        # 1. Convert the 'label' column to a list of integers (the true labels)
        true_labels = df['label'].astype(int).tolist()
        
        # 2. Assume the length classification column is named 'is_short' or 'is_long' in the CSV.
        # Since your sample header has 'is_short' and 'is_long' is not in the sample, 
        # we will use a derived column based on the example structure.
        # The true indicator of length in your sample is the 'is_short' column (True/False).
        # We'll create a 'length_group' based on it.
        if 'is_short' not in df.columns:
            print("WARNING: 'is_short' column not found. Calculating TOTAL accuracy only.")
            length_groups = None
        else:
            # Map 'True' to 'is_short' group, and everything else to 'is_long' group
            # (Assuming a story is not 'is_short' means it is 'is_long')
            df['length_group'] = df['is_short'].apply(lambda x: 'is_short' if x in [True, 'True'] else 'is_long')
            length_groups = df['length_group'].tolist()

        
        # --- MAKE PREDICTIONS ---
        print("Making predictions on the test data...")
        # Apply the prediction function to every text input
        predicted_labels = [predict_moral_status(text) for text in df['input']]
        df['predicted_label'] = predicted_labels
        
        # Remove any rows where prediction failed (e.g., if model load was partially successful)
        df_clean = df.dropna(subset=['predicted_label'])
        true_labels_clean = df_clean['label'].astype(int).tolist()
        predicted_labels_clean = df_clean['predicted_label'].astype(int).tolist()
        
        if not true_labels_clean:
            print("ERROR: No valid predictions were made.")
        else:
            # --- CALCULATE TOTAL ACCURACY ---
            total_accuracy = accuracy_score(true_labels_clean, predicted_labels_clean)

            # --- CALCULATE GROUP ACCURACIES ---
            if length_groups is not None:
                print("\n" + "="*40)
                print("       🌟 MORAL CLASSIFIER ACCURACY 🌟")
                print("="*40)
                
                # Accuracy for 'is_short'
                short_df = df_clean[df_clean['length_group'] == 'is_short']
                if not short_df.empty:
                    short_acc = accuracy_score(short_df['label'], short_df['predicted_label'])
                    print(f"Accuracy for 'is_short' stories: {short_acc:.4f} ({len(short_df)} samples)")
                
                # Accuracy for 'is_long'
                long_df = df_clean[df_clean['length_group'] == 'is_long']
                if not long_df.empty:
                    long_acc = accuracy_score(long_df['label'], long_df['predicted_label'])
                    print(f"Accuracy for 'is_long' stories:  {long_acc:.4f} ({len(long_df)} samples)")
                
                print("-" * 40)
                
            print(f"TOTAL ACCURACY (All samples):     {total_accuracy:.4f} ({len(true_labels_clean)} samples)")
            print("="*40 + "\n")

    except FileNotFoundError:
        print(f"ERROR: The file '{TEST_CSV_PATH}' was not found. Please check the path.")
    except KeyError as e:
        print(f"ERROR: Missing expected column in CSV: {e}. Ensure the file has 'label' and 'input' columns.")