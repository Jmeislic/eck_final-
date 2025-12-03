## To create this codebase I asked Google Gemini "Can you write me a python file which takes in a csv at file path ".\moral_stories_csv\data_classification_action+context+consequence_lexical_bias_train.csv", and goes through each row of the dataset, taking in each item in the row (minus label), and then using that information and a query BART Model constructs a sentence that turns the different information into one sentence explaining the moral situation. The program then writes the row into a new csv file called data_With_Sentence, with a new collum labeled explanation" which contains the created sentence."

import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import os
import time

# --- Configuration ---
INPUT_FILEPATH = "./moral_stories_csv/data_classification_action+context+consequence_lexical_bias_train.csv"
OUTPUT_FILEPATH = "data_With_Sentence.csv"
MODEL_NAME = "facebook/bart-large-cnn" # Excellent for summarization/generation
# The columns from the CSV that will be combined to form the input prompt (excluding 'label')
INPUT_COLUMNS = ["norm","situation","intention","moral_action","moral_consequence","label","immoral_action","immoral_consequence"] 

# --- Environment Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Initialize Model and Tokenizer ---
try:
    print(f"Loading BART model: {MODEL_NAME}...")
    # Suppress a potential warning about a missing tokenizer file if the model is downloaded for the first time
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    print("BART model loaded successfully.")
    
    # Flag to check if model loading was successful
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading model or tokenizer. Please ensure 'torch' and 'transformers' are installed correctly.")
    print(f"Details: {e}")
    MODEL_LOADED = False



# --- Core Sentence Generation Function ---
def generate_moral_sentence(row):
    print(f"row= {row}")
    """
    Combines story components, queries the BART model, and returns 
    a single explanatory sentence.
    """
    if not MODEL_LOADED:
        return "ERROR: Model not loaded."
        
    try:
        # 1. Combine inputs into a prompt string
        input_parts = [f"{col}: {row[col]}" for col in INPUT_COLUMNS if col in row and pd.notna(row[col])]
        input_text = " ".join(input_parts)
        
        # 2. Tokenize and prepare input
        inputs = tokenizer(
            input_text, 
            max_length=1024, 
            return_tensors='pt', 
            truncation=True
        ).to(device)

        # 3. Generate the summary/sentence using exponential backoff retry logic
        # This handles potential temporary GPU/memory issues or resource contention
        max_retries = 3
        delay = 1 # seconds
        for attempt in range(max_retries):
            try:
                summary_ids = model.generate(
                    inputs['input_ids'],
                    num_beams=4,        # Use beam search for higher quality
                    min_length=30,      # Ensure a meaningful sentence is generated
                    max_length=100,
                    length_penalty=2.0,
                    early_stopping=True, # Stop when all beams have finished
                    no_repeat_ngram_size=3
                )

                # 4. Decode the result
                generated_sentence = tokenizer.decode(
                    summary_ids.squeeze(), 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )
                
                # Cleanup: Ensure it is a single, clear sentence
                if '.' in generated_sentence:
                    generated_sentence = generated_sentence.split('.')[0] + '.'
                print(f"This is the sentence = {generated_sentence.strip()}")
                print("")
                return generated_sentence.strip()

            except RuntimeError as e:
                # Catch CUDA out-of-memory or other runtime errors
                if attempt < max_retries - 1:
                    print(f"Runtime error on row {row.name}. Retrying in {delay}s... ({e})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"Failed to generate sentence for row {row.name} after {max_retries} attempts.")
                    return f"ERROR: Generation failed after {max_retries} retries."

    except KeyError as e:
        print(f"Skipping row {row.name}: Missing expected column data ({e}).")
        return "ERROR: Missing column data."
    except Exception as e:
        print(f"An unexpected error occurred during generation for row {row.name}: {e}")
        return "ERROR: Unknown exception during generation."


# --- Main Execution Function ---
def process_data_with_bart():
    if not MODEL_LOADED:
        return

    print(f"\nStarting data processing from {INPUT_FILEPATH}...")
    
    # Check if the input file exists and create a mock if it does not
    if not os.path.exists(INPUT_FILEPATH):
        print("Error No file")
    # 1. Load the dataset
    try:
        df = pd.read_csv(INPUT_FILEPATH)
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return

    # Check for required columns
    if not all(col in df.columns for col in INPUT_COLUMNS) or 'label' not in df.columns:
        print(f"Error: CSV is missing one of the required columns ({INPUT_COLUMNS} and 'label').")
        print(f"Found columns: {list(df.columns)}")
        print("Please verify your CSV structure.")
        return

    # 2. Apply the generation function to each row
    print(f"Generating explanatory sentences for {len(df)} rows...")
    df['explanation'] = df.apply(generate_moral_sentence, axis=1)

    # 3. Save the new DataFrame
    try:
        df.to_csv(OUTPUT_FILEPATH, index=False)
        print("\n" + "=" * 60)
        print(f"Processing complete! The updated data has been saved to: {OUTPUT_FILEPATH}")
        print(f"A new column 'explanation' was added.")
        print("=" * 60)
        print("\nFirst 5 rows of the output data:")
        print(df.head())
    except Exception as e:
        print(f"Failed to save output CSV file: {e}")

# Execute the main function
if __name__ == "__main__":
    process_data_with_bart()