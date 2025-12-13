import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
# Import BART components for the decomposition task
from transformers import BartTokenizer, BartForConditionalGeneration 
import numpy as np
import re # Import regex for parsing BART output

# --- CONFIGURATION (Based on your provided script) ---
# BERT CLASSIFIER CONFIG
TEST_CSV_PATH = "testingSets/dataSet_1_test.csv"
MORAL_MODEL_PATH = "./bert_moral_classifier"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

# BART DECOMPOSITION CONFIG
BART_MODEL_PATH = "./bertParsed2"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
ID_TO_LABEL = {0: 'IMMORAL', 1: 'MORAL'} 

# --- LOAD MORAL CLASSIFIER ---
moral_tokenizer = None
moral_model = None
try:
    print(f"Loading moral classifier from {MORAL_MODEL_PATH}...")
    moral_tokenizer = BertTokenizerFast.from_pretrained(MORAL_MODEL_PATH)
    moral_model = BertForSequenceClassification.from_pretrained(MORAL_MODEL_PATH)
    moral_model.to(DEVICE)
    moral_model.eval()
except Exception as e:
    print(f"FATAL ERROR: Could not load moral classifier. Error: {e}")

# --- LOAD BART DECOMPOSITION MODEL ---
bart_tokenizer = None
bart_model = None
try:
    print(f"Loading BART decomposition model from {BART_MODEL_PATH}...")
    bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_PATH)
    bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_PATH)
    bart_model.to(DEVICE)
    bart_model.eval()
except Exception as e:
    print(f"FATAL ERROR: Could not load BART decomposition model. Error: {e}")

# --- BART INFERENCE FUNCTION (Copied from your script) ---
def generate_decomposition(input_sentence: str) -> str:
    """Loads the fine-tuned BART model and generates the decomposed output."""
    if bart_model is None or bart_tokenizer is None:
        return "norm: Error situation: Error intention: Error action: Error"
    
    try:
        # Tokenize the input sentence
        inputs = bart_tokenizer(
            [input_sentence], 
            max_length=MAX_INPUT_LENGTH, 
            return_tensors="pt", 
            truncation=True
        ).to(DEVICE)

        # Generate the output sequence
        output_ids = bart_model.generate(
            inputs["input_ids"], 
            max_length=MAX_TARGET_LENGTH, 
            num_beams=4,
            early_stopping=True
        )

        # Decode and return the generated text
        output_text = bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    except Exception as e:
        #f"Error during BART generation: {e}")
        return "norm: Error situation: Error intention: Error action: Error"

import re

# --- HELPER: Parse BART's structured output into a dictionary (UPDATED) ---
def parse_bart_output(bart_output: str) -> dict:
    """
    Parses the BART output string (e.g., 'norm: text situation: text...') 
    into a dictionary by looking for specific field names.
    """
    spans = {"norm": "", "situation": "", "intention": "", "action": ""}
    
    # 1. Normalize the BART output for consistent processing (ensure all keys are lowercased and single-spaced)
    normalized_output = bart_output.lower().strip()
    
    # 2. Define the keys we are looking for in the specific order
    keys = ["norm", "situation", "intention", "action"]
    
    # 3. Create a combined regex pattern that looks for the keys and captures the content
    # This pattern captures all text non-greedily (.*?) after a key: until the next key: or the end of the string.
    pattern = r"(norm|situation|intention|action):\s*(.*?)(?=\s*(?:situation|intention|action):|$)"
    
    matches = re.findall(pattern, normalized_output, re.IGNORECASE)
    
    # 4. Populate the dictionary with extracted components
    for key, content in matches:
        # Use the key as the dictionary key (e.g., 'norm', 'situation')
        spans[key.strip()] = content.strip()
        
    # Check if a component was missed (e.g., the last one) and try to catch it if possible
    # This is a fallback check for the last component if the lookahead failed
    if not spans['action'] and 'action:' in normalized_output:
        # Simple fallback for the last element
        parts = normalized_output.split('action:')
        if len(parts) > 1:
            spans['action'] = parts[-1].strip()

    return spans


# --- CLASSIFICATION FUNCTION ---
def predict_moral_status_short(text):
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

    moral_status = ID_TO_LABEL.get(pred_id, "UNKNOWN")
    return pred_id

# --- MAIN PREDICTION FUNCTION (Uses BART for Extraction) ---
def predict_moral_status(text):
    if len(text)>30:
        """Performs component extraction (BART) and moral classification (BERT)."""
        
        # --- 1. Component Extraction using BART (Seq2Seq) ---

        bart_output_text = generate_decomposition(text)
        spans = parse_bart_output(bart_output_text)
        
        # --- 2. Moral Classification using BERT (Sequence Classification) ---

        if moral_model is None:
            moral_status = "UNKNOWN (Classifier Missing)"
        else:
            # BERT classifier was trained on the combination of elements
            combined_text = " ".join([spans.get(k, "") for k in ['norm', 'situation', 'intention', 'action'] if spans.get(k)])
            
            # Fallback if BART failed to extract components
            if combined_text.strip() == "":
                combined_text = text 
            
            enc2 = moral_tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                out2 = moral_model(**enc2)
                
            pred_label = torch.argmax(out2.logits, dim=-1).item()
            moral_status = ID_TO_LABEL.get(pred_label, "UNKNOWN")


        norm = spans.get("norm", "")
        sit = spans.get("situation", "")
        intent = spans.get("intention", "")
        action = spans.get("action", "")
        
        
    else:
        pred_label=predict_moral_status_short(text)
        norm = ""
        sit =""
        intent = ""
        action = ""

    return pred_label


## ⭐️ Crucial Data Transformation Function ⭐️
def transform_moral_stories_to_classification(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the Moral Stories dataset format (one row = two branches)
    into a classification format (two rows = two labeled stories) suitable for evaluation.
    This creates the required 'input' and 'label' columns.
    """
    new_data = []

    # Columns containing the components
    components = ['norm', 'situation', 'intention']
    
    for _, row in df_original.iterrows():
        # --- 1. Create the MORAL story row (Label 1) ---
        moral_story_text = " ".join([
            row.get(col, '') for col in components
        ]) + " " + row.get('moral_action', '') + " " + row.get('moral_consequence', '')
        
        new_data.append({
            'input': moral_story_text.strip(),
            'label': 1,
            # 'is_short' is derived from presence of other fields in your data if needed, 
            # but for now we'll assume it's missing or derived later.
        })

        # --- 2. Create the IMMORAL story row (Label 0) ---
        immoral_story_text = " ".join([
            row.get(col, '') for col in components
        ]) + " " + row.get('immoral_action', '') + " " + row.get('immoral_consequence', '')
        
        new_data.append({
            'input': immoral_story_text.strip(),
            'label': 0,
        })

    return pd.DataFrame(new_data)


# --- MAIN EVALUATION LOGIC ---
if moral_model is not None:
    #"\n" + "="*50)
    #"🚀 Starting Model Evaluation & Accuracy Calculation 🚀")
    #"="*50)
    
    #f"Reading test data from {TEST_CSV_PATH}...")
    try:
        # 1. Load the raw data
        df_raw = pd.read_csv(TEST_CSV_PATH)
        
        # 2. TRANSFORM the data to the expected 'input' and 'label' format
        df = transform_moral_stories_to_classification(df_raw) 

        # --- PREPARE DATA ---
        true_labels = df['label'].astype(int).tolist()
        
        # --- MAKE PREDICTIONS ---
        # Note: We use .apply(lambda x: ...) for better readability, 
        # but a list comprehension is often faster if needed.
        #"Making predictions on the test data...")
        df['predicted_label'] = df['input'].apply(predict_moral_status)
        
        # Remove any rows where prediction failed (e.g., if model load was partially successful)
        df_clean = df.dropna(subset=['predicted_label'])
        true_labels_clean = df_clean['label'].astype(int).tolist()
        predicted_labels_clean = df_clean['predicted_label'].astype(int).tolist()
        
        if not true_labels_clean:
            print("ERROR: No valid predictions were made.")
        else:
            # --- CALCULATE TOTAL ACCURACY ---
            total_accuracy = predicted_labels_clean/true_labels_clean

            #"\n" + "="*50)
            #"                 🌟 FINAL ACCURACY 🌟")
            #"="*50)
            print(f"TOTAL ACCURACY (Evaluated samples):   {total_accuracy:.4f} ({len(true_labels_clean)} samples)")
            #"="*50 + "\n")

    except FileNotFoundError:
        print(f"ERROR: The file '{TEST_CSV_PATH}' was not found. Please check the path.")
    except KeyError as e:
        print(f"ERROR: Missing expected column in CSV: {e}. Ensure the raw file has all Moral Stories components.")

# --- Interactive Mode (for testing individual sentences) ---
if __name__ == '__main__':
    # We redefine predict_moral_status for interactive use to include printing
    def interactive_predict_moral_status(text):
        if moral_model is None:
            #"Error: Moral classifier not loaded.")
            return

        # 1. Component Extraction
        #"\n--- 1. Running BART for Component Extraction ---")
        bart_output_text = generate_decomposition(text)
        spans = parse_bart_output(bart_output_text)
        
        # 2. Moral Classification
        combined_text = " ".join([spans.get(k, "") for k in ['norm', 'situation', 'intention', 'action'] if spans.get(k)])
        if combined_text.strip() == "":
            combined_text = text 
        
        enc2 = moral_tokenizer(combined_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            out2 = moral_model(**enc2)
        pred_label = torch.argmax(out2.logits, dim=-1).item()
        moral_status = ID_TO_LABEL.get(pred_label, "UNKNOWN")

        # 3. Print Results
        print("\n--- Results ---")
        print(f"BART Raw Output: {bart_output_text}")
        print("-" * 30)
        print(f"NORM:      {spans.get('norm', '')}")
        print(f"SITUATION: {spans.get('situation', '')}")
        print(f"INTENTION: {spans.get('intention', '')}")
        print(f"ACTION:    {spans.get('action', '')}")
        print("-" * 30)
        print(f"\nMoral Status (BERT): {moral_status}")

    while True:
        test_sentence = input("\nEnter a sentence for moral analysis (or type 'quit'): ")
        if test_sentence.lower() == 'quit':
            break
        interactive_predict_moral_status(test_sentence)