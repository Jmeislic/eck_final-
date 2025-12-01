import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
import os
import string # Required for component cleaning
from bart_explanation import run_moral_explanation

# -------------------------
# CONFIG
# -------------------------
COMPONENT_MODEL_PATH = "./bert_moral_components"
MORAL_MODEL_PATH = "./bert_moral_classifier"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3

# -------------------------
# HELPER FUNCTIONS FOR DATA PREPARATION (The Fix)
# -------------------------
def clean_component_text(text):
    """
    Strips common component delimiters (like periods, commas, colons, and quotes) 
    from the ends of the text to prevent the model from learning these as 
    segmentation markers.
    """
    if pd.isna(text) or text.strip() == "":
        return ""
    
    cleaned_text = text.strip()
    
    # 1. Strip trailing punctuation
    trailing_punct = {'.', ',', ';', ':', '?', '!', '"', "'"}
    while cleaned_text and cleaned_text[-1] in trailing_punct:
        cleaned_text = cleaned_text[:-1].strip()
        
    # 2. Strip leading punctuation
    leading_punct = {':', '"', "'"}
    while cleaned_text and cleaned_text[0] in leading_punct:
        cleaned_text = cleaned_text[1:].strip()
        
    return cleaned_text

def create_text_and_offsets(row):
    """
    Combines story components into one text string and calculates the 
    character-level offsets for each component.
    """
    components = []
    current_char_offset = 0
    full_text = ""
    
    for label_type in ["SITUATION", "NORM", "INTENTION", "ACTION"]:
        
        # Apply the cleaning step here
        raw_text = row.get(label_type.lower(), '')
        comp_text = clean_component_text(raw_text) 
        
        if not comp_text:
            continue

        # Add a single space to separate components
        if full_text:
            full_text += " "
            current_char_offset += 1 

        start_char = current_char_offset
        end_char = start_char + len(comp_text)
        
        components.append({
            'label': label_type,
            'start': start_char,
            'end': end_char
        })
        
        full_text += comp_text
        current_char_offset = end_char
        
    return full_text, components

# -------------------------
# LOAD DATASET
# -------------------------
df = pd.read_csv("moral_stories_csv/data_moral_stories_full.csv")

# -------------------------
# PREPARE TOKEN LABELS
# -------------------------
labels = ["O", "B-SITUATION", "I-SITUATION", "B-NORM", "I-NORM", "B-INTENTION", "I-INTENTION", "B-ACTION", "I-ACTION"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# -------------------------
# TOKENIZATION DATASET
# -------------------------
tokenizer_for_dataset = BertTokenizerFast.from_pretrained("bert-base-uncased")

class MoralComponentsDataset(Dataset):
    def __init__(self, dataframe):
        self.examples = []
        
        for idx, row in dataframe.iterrows():
            
            # 1. Get the combined text and character offsets for components (using the CLEANED function)
            full_text, component_offsets = create_text_and_offsets(row)
            
            if not full_text:
                continue

            # 2. Tokenize and request offset mapping
            enc = tokenizer_for_dataset(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
                return_offsets_mapping=True # CRITICAL FOR ALIGNMENT
            )
            
            offsets = enc.pop("offset_mapping").squeeze(0) # Character start/end for each token
            token_labels = [-100] * MAX_LEN

            # 3. Align labels using character offsets
            for i, (token_start, token_end) in enumerate(offsets.tolist()):
                if enc["input_ids"][0, i] in tokenizer_for_dataset.all_special_ids:
                    continue # Skip special tokens [CLS], [SEP], [PAD]

                token_label_set = False
                for comp in component_offsets:
                    comp_start = comp['start']
                    comp_end = comp['end']
                    label_key = comp['label']
                    
                    # Check if the token is *inside* the component span
                    if token_start >= comp_start and token_end <= comp_end:
                        
                        # B- Tag: If the token starts exactly at the component's start
                        if token_start == comp_start:
                            token_labels[i] = label2id[f"B-{label_key}"]
                        # I- Tag: All other tokens inside the span
                        else:
                            token_labels[i] = label2id[f"I-{label_key}"]
                        
                        token_label_set = True
                        break
                
                # If the token wasn't matched to any component, it's 'O' (Outside)
                if not token_label_set:
                    token_labels[i] = label2id["O"]
            
            enc['labels'] = torch.tensor([token_labels])
            # Prepare the example dictionary for the dataset
            self.examples.append({k: v.squeeze(0) for k, v in enc.items() if k != 'offset_mapping'})
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = MoralComponentsDataset(df)

# -------------------------
# TRAIN TOKEN-CLASSIFIER (Component Model)
# -------------------------
if os.path.isdir(COMPONENT_MODEL_PATH):
    print(f"✅ Loading saved component model from {COMPONENT_MODEL_PATH}...")
    tokenizer = BertTokenizerFast.from_pretrained(COMPONENT_MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(COMPONENT_MODEL_PATH)
    model.to(DEVICE)
else:
    print(f"❌ Saved component model not found. Training new model and saving to {COMPONENT_MODEL_PATH}...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased") 
    
    # Ensure label mapping is in model config for proper saving/loading
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels), 
        id2label=id2label,
        label2id=label2id
    )
    
    model.to(DEVICE)
    training_args = TrainingArguments(
        output_dir=COMPONENT_MODEL_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained(COMPONENT_MODEL_PATH)
    tokenizer.save_pretrained(COMPONENT_MODEL_PATH)


# -------------------------
# LOAD MORAL CLASSIFIER
# -------------------------
# NOTE: This assumes you have a pre-trained moral classifier model at MORAL_MODEL_PATH
try:
    moral_tokenizer = BertTokenizerFast.from_pretrained(MORAL_MODEL_PATH)
    moral_model = BertForSequenceClassification.from_pretrained(MORAL_MODEL_PATH)
    moral_model.to(DEVICE)
    moral_model.eval()
except Exception as e:
    print(f"⚠️ Could not load moral classifier from {MORAL_MODEL_PATH}. Check if model exists.")
    moral_model = None


# -------------------------
# HELPER FUNCTIONS FOR INFERENCE
# -------------------------
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
                spans[current_label] += " " + tokenizer.convert_tokens_to_string(span_tokens)
            
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
                spans[current_label] += " " + tokenizer.convert_tokens_to_string(span_tokens)
            current_label = None
            span_tokens = []
            
    # Capture the last remaining span
    if current_label and span_tokens:
        spans[current_label] += " " + tokenizer.convert_tokens_to_string(span_tokens)
        
    for k in spans:
        spans[k] = spans[k].strip()
        
    return spans


def classify_moral_action(text):
    """Performs component extraction and moral classification."""
    if moral_model is None:
        return {"SITUATION": "Model Error", "NORM": "Model Error", "INTENTION": "Model Error", "ACTION": "Model Error"}, "UNKNOWN (Classifier Missing)"

    # Extract components
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    with torch.no_grad():
        outputs = model(**enc)
    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
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
    
    return spans, moral_status

# -------------------------
# PROMPT USER INPUT
# -------------------------
if __name__ == "__main__":
    while True:
        user_input = input("Input: ")
        if user_input.lower() == "exit":
            break
        spans, status = classify_moral_action(user_input)
        print("\nExtracted components:")
        for k, v in spans.items():
            print(f"{k}: {v}")
        print(f"\nPredicted moral status: {status}")
        print(f"\nExplaination: {run_moral_explanation()}")