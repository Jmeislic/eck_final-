#This top section was copied from logic.py

import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
# Import BART components for the decomposition task
from transformers import BartTokenizer, BartForConditionalGeneration 
import numpy as np
import re # Import regex for parsing BART output
import csv
import matplotlib.pyplot as plt

# from makeBert2_2 import predict_moral_status
# --- CONFIGURATION (Based on your provided script) ---
# BERT CLASSIFIER CONFIG
MORAL_MODEL_PATH = "./bert_moral_classifier_split_no_consiquence"
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


def predict_moral_status(text):
    if len(text)>30:
        """Performs component extraction (BART) and moral classification (BERT)."""
        
        # --- 1. Component Extraction using BART (Seq2Seq) ---
        # print("\n--- 1. Running BART for Component Extraction ---")
        bart_output_text = generate_decomposition(text)
        spans = parse_bart_output(bart_output_text)
        
        # --- 2. Moral Classification using BERT (Sequence Classification) ---
        # print("\n--- 2. Running BERT for Moral Classification ---")
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

        # print("\n--- Results ---")
        # for k, v in spans.items():
        #     print(f"{k.upper()}: {v}")
        norm = spans.get("norm", "")
        sit = spans.get("situation", "")
        intent = spans.get("intention", "")
        action = spans.get("action", "")
        
        
    else:
        moral_status=predict_moral_status_short(text)
        norm = ""
        sit =""
        intent = ""
        action = ""
    # print(f"\nMoral Status (BERT): {moral_status}")
    return moral_status, norm, sit, intent, action
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
        print(f"Error during BART generation: {e}")
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
    # print(f"is pred_id {pred_id}")
    moral_status = ID_TO_LABEL.get(pred_id, "UNKNOWN")
    return moral_status

#This equation was taken from the slides on evaluting classifcation 
def makeConfidenceInterval (n, accuracy):
    SE = np.sqrt(((accuracy*(1-accuracy))/n))
    Z = 1.96
    low = accuracy - (Z*SE)
    high = accuracy + (Z*SE)
    return(low, high)




# To make this function I asked ChatGPT "Can you make me a python function which takes in four accuracys for, "Main Test", "Total Second", "Short Second", "Long Second", and creates a graph of those four values."

def plot_accuracies(main_test, total_second, short_second, long_second):
    labels = [
        "Main Test",
        "Total Second",
        "Short Second",
        "Long Second"
    ]
    
    accuracies = [
        main_test,
        total_second,
        short_second,
        long_second
    ]
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, accuracies)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    
    for i, value in enumerate(accuracies):
        plt.text(i, value + 1, f"{value}%", ha="center")
    
    plt.tight_layout()
    plt.show()


# print(predict_moral_status("I stole a car from a baby because I am evil")[0])

# To create this code below I asked ChatGPT "Can you create me a Python file that uses the function predict_moral_status(), which takes in as input a string, runs the function on each line of the CSV dataset_1_test.csv, this CSV has 3 columns, which are important: label, input, and is_short. Run the input through the predict function, if the output matches the label output it is a sucess, if it does not it is a failure. Please keep track of short sentences (when is_short is True), and long sentences (when is_short is False) and a total accuracy which is both. Please then print these outputs"
total = 0
total_correct = 0

short_total = 0
short_correct = 0

long_total = 0
long_correct = 0
n1 = 0
with open("./final_test.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        n1 += 1
        if total%100 == 0:
            print(f"At row: {total}")
        label = row["label"]
        text_input = row["explanation"]
        # is_short = row["is_short"].lower() == "true"

        prediction = predict_moral_status(text_input)[0]

        # print(prediction)
        # print(label)

        total += 1
        # if is_short:
        #     short_total += 1
        # else:
        #     long_total += 1
        if label == "1":
            if prediction == "MORAL":
                # print("Was correct")
                total_correct += 1
                # if is_short:
                #     short_correct += 1
                # else:
                #     long_correct += 1
        if label == "0":
            if prediction == "IMMORAL":
                # print("was correct immoral")
                total_correct += 1
                # if is_short:
                #     short_correct += 1
                # else:
                #     long_correct += 1
                

total2 = 0
total_correct2 = 0
n2 = 0
with open("./testingSets/dataSet_1_test.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        n2 += 1
        if total%100 == 0:
            print(f"At row: {total} of dataset 2")
        label = row["label"]
        text_input = row["input"]
        is_short = row["is_short"].lower() == "true"
        prediction = predict_moral_status(text_input)[0]

        # print(prediction)
        # print(label)

        total2 += 1
        if is_short:
            short_total += 1
        else:
            long_total += 1
        if label == "0":
            if prediction == "MORAL":
                # print("Was correct")
                total_correct2 += 1
                if is_short:
                    short_correct += 1
                else:
                    long_correct += 1
        if label == "1":
            if prediction == "IMMORAL":
                # print("was correct immoral")
                total_correct2 += 1
                if is_short:
                    short_correct += 1
                else:
                    long_correct += 1

# Avoid division by zero
total_accuracy = total_correct / total if total > 0 else 0.0
total_accuracy2 = total_correct2 / total2 if total2 > 0 else 0.0
short_accuracy = short_correct / short_total if short_total > 0 else 0.0
long_accuracy = long_correct / long_total if long_total > 0 else 0.0

print("Evaluation Results Main Test")
print("-" * 30)
print(f"Total samples: {total}")
print(f"Total accuracy: {total_accuracy:.4f}")
print(f"The confidence interval is {makeConfidenceInterval(n1, total_accuracy)}")
print()
print("This is for the second test set:")
print(f"Total samples: {total2}")
print(f"Total accuracy: {total_accuracy2:.4f}")
print(f"The confidence interval is {makeConfidenceInterval(n2, total_accuracy2)}")
print()
print(f"Short sentences: {short_total}")
print(f"Short sentence accuracy: {short_accuracy:.4f}")
print(f"The confidence interval is {makeConfidenceInterval(short_total, short_accuracy)}")
print()
print(f"Long sentences: {long_total}")
print(f"Long sentence accuracy: {long_accuracy:.4f}")
print(f"The confidence interval is {makeConfidenceInterval(long_total, long_accuracy)}")
plot_accuracies(total_accuracy, total_accuracy2, short_accuracy, long_accuracy)
