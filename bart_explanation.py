#To start off this code base I asked Google Gemini "Can you create for me a python program which trains a custom instance of the AI model BART, which creates a sentence describing why an action was moral or immoral from csv. The csv contains 8 collums, "norm" "situation" "intention" "moral_action" "moral_consequence" "immoral_action" "immoral_consequence" and "label". The output from this model should an explanation for why an action is either moral or immoral depending on the label, where a moral action is a label of 0 and an immoral action is a label of 1. If the action is moral, then there will be a moral_action and moral_consequence, if the action is immoral there will be an immoral_action and immoral_consequence, sometimes there will be both for the same sentence. When this model is run it will receive an input sentence and a decision if the sentence is moral or immoral in binary form. Please create for me code which creates a model which can then by run by a function call with the two inputs specified above."

import pandas as pd
from datasets import Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch
import os

# --- Configuration ---
CSV_FILE_PATH = 'moral_stories_csv/data_classification_action+context+consequence_lexical_bias_train.csv' # CHANGE THIS to your actual CSV file name
MODEL_NAME = "facebook/bart-base" # Using 'bart-base' for quicker fine-tuning
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
OUTPUT_DIR = "./bart_moral_reasoning_model"

# --- 1. Data Preparation Function ---

def prepare_dataset_for_bart(df: pd.DataFrame, tokenizer: BartTokenizer) -> Dataset:
    """
    Transforms the DataFrame into input prompts (for BART) and target explanations (labels).
    """
    processed_data = []

    for index, row in df.iterrows():
        # Step 1: Determine the action/consequence and the target explanation (output)
        label = row['label']
        norm = row['norm']
        situation = row['situation']
        intention = row['intention']
        
        if label == 0:  # Moral Action
            action = row['moral_action']
            consequence = row['moral_consequence']
            # Target explanation (You must have this column in your real CSV for training!)
            # ***NOTE: You MUST include a column in your CSV, e.g., 'explanation', 
            # with the ground-truth text explaining the moral decision.***
            target_explanation = row.get('moral_explanation', "The action aligns with the norm.")
            judgment = "Moral"
        elif label == 1: # Immoral Action
            action = row['immoral_action']
            consequence = row['immoral_consequence']
            target_explanation = row.get('immoral_explanation', "The action violates the norm.")
            judgment = "Immoral"
        else:
            continue

        # Step 2: Create the input prompt (source text)
        input_prompt = (
            f"The situation is: {situation}. The agent's intention was: {intention}. "
            f"The action taken was: {action}. The consequence was: {consequence}. "
            f"The relevant norm is: {norm}. The final judgment is {judgment}. "
            "Explain the reason for this judgment in one sentence."
        )

        processed_data.append({
            'input_text': input_prompt,
            'target_text': target_explanation
        })

    # Step 3: Tokenization
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input_text'], max_length=MAX_INPUT_LENGTH, truncation=True
        )
        labels = tokenizer(
            examples['target_text'], max_length=MAX_TARGET_LENGTH, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert list of dicts to Hugging Face Dataset
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
    return tokenized_datasets

# --- 2. Main Training Function ---

def train_bart_model():
    """
    Loads data, sets up the BART model, and initiates the fine-tuning process.
    """
    if not os.path.exists(CSV_FILE_PATH):
        print(f"ERROR: CSV file not found at {CSV_FILE_PATH}.")
        print("Please create a CSV file with your data and a column named 'moral_action' or 'immoral_action'.")
        # Creating a dummy file for demonstration structure
        dummy_df = pd.DataFrame({
            "norm": ["Don't lie"], "situation": ["A lie is told"], "intention": ["to deceive"], 
            "moral_action": ["truth"], "moral_consequence": ["good"], 
            "immoral_action": ["lie"], "immoral_consequence": ["bad"], "label": [1], 
            "moral_action": ["explanation 0"], "immoral_action": ["explanation 1"] 
        })
        dummy_df.to_csv(CSV_FILE_PATH, index=False)
        print(f"Created a dummy CSV at {CSV_FILE_PATH}. Please populate it with your training data.")
        return None

    df = pd.read_csv(CSV_FILE_PATH)
    
    # Check if a 'ground truth' explanation column exists (essential for training)
    if 'moral_action' not in df.columns and 'immoral_action' not in df.columns:
        print("\nFATAL ERROR: Training requires a 'ground truth' explanation column.")
        print("Please add a column like 'explanation' to your CSV containing the correct moral reasoning for the model to learn.")
        return None
        
    print(f"--- Loaded {len(df)} data samples for fine-tuning ---")

    # Initialize Tokenizer and Model
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Prepare Data
    tokenized_dataset = prepare_dataset_for_bart(df, tokenizer)
    
    # Split for Training/Evaluation (80/20 split)
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                # Number of training epochs
        per_device_train_batch_size=4,     # Batch size per device (adjust based on GPU memory)
        per_device_eval_batch_size=4,
        warmup_steps=500,                  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                 # Strength of weight decay
        logging_dir='./logs',              # Directory for storing logs
        logging_steps=100,
        evaluation_strategy="epoch",       # Evaluate after each epoch
        save_strategy="epoch",             # Save a checkpoint after each epoch
        load_best_model_at_end=True,       # Load the best model found during training
        predict_with_generate=True,        # Use generation for prediction
        fp16=torch.cuda.is_available()     # Use 16-bit precision if a GPU is available
    )

    # Initialize Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start Training
    print("\n--- Starting BART Fine-Tuning (Requires GPU/Time) ---")
    trainer.train()
    
    # Save the final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n--- Fine-tuning complete. Model saved to {OUTPUT_DIR} ---")
    
    return BartForConditionalGeneration.from_pretrained(OUTPUT_DIR), BartTokenizer.from_pretrained(OUTPUT_DIR)


# --- 3. Prediction Function (How to Run the Trained Model) ---

def run_moral_explanation(model, tokenizer, input_sentence: str, label: int) -> str:
    """
    Generates an explanation using the fine-tuned BART model.
    This function meets your final requirement for a callable model.
    """
    if model is None or tokenizer is None:
        return "Model not available. Run the training script first."
    
    # Map the label to a readable judgment
    moral_judgment = "Moral" if label == 0 else "Immoral"

    # Reconstruct the exact prompt used during training!
    # The action, consequence, norm, and intention must be part of the input_sentence
    final_prompt = (
        f"{input_sentence} The final judgment is {moral_judgment}. "
        "Explain the reason for this judgment in one sentence."
    )
    final_prompt = " ".join(final_prompt.split())
    
    # Tokenize the input
    input_ids = tokenizer.encode(
        final_prompt,
        return_tensors='pt',
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    ).to(model.device) # Move to the same device as the model

    # Generate the explanation
    outputs = model.generate(
        input_ids,
        max_length=MAX_TARGET_LENGTH,
        min_length=10,
        num_beams=4,
        do_sample=False,
        early_stopping=True
    )

    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return explanation.strip()

# --- 4. Execution Block ---

if __name__ == "__main__":
    # Attempt to train the model and get the trained instance
    # This section needs to be uncommented and run ONCE to fine-tune the model.
    trained_model, trained_tokenizer = train_bart_model() 
    
    # --- DEMONSTRATION OF MODEL USAGE (After Training) ---
    
    # Load the trained model from the saved directory (assuming training was run successfully)
    
    # try:
    #     if os.path.exists(OUTPUT_DIR):
    #         trained_model = BartForConditionalGeneration.from_pretrained(OUTPUT_DIR)
    #         trained_tokenizer = BartTokenizer.from_pretrained(OUTPUT_DIR)
    #     else:
    #         # Fallback to a pre-trained model for demonstration if training hasn't run
    #         print("WARNING: Using a pre-trained BART model for demonstration (not fine-tuned).")
    #         trained_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    #         trained_tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        
    #     # Move model to device
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     trained_model.to(device)
    #     print(f"Model loaded for inference on device: {device}")

    #     # Example Inputs (must be similar to the training prompt format)
    #     example_context_moral = (
    #         "The situation is: A child finds a wallet on the street. The agent's intention was: to return the wallet to its owner. "
    #         "The action taken was: giving the wallet to the police. The consequence was: the owner got the wallet back. "
    #         "The relevant norm is: Always be honest and respect others' property."
    #     )
        
    #     example_context_immoral = (
    #         "The situation is: A server mistakenly adds an extra $\$50$ to your bill. The agent's intention was: to get free money. "
    #         "The action taken was: quietly paying the bill and saying nothing. The consequence was: the restaurant lost money. "
    #         "The relevant norm is: Do not exploit others' mistakes for personal gain."
    #     )

    #     print("\n--- Model Inference ---")
        
    #     # Call the function for a Moral action (Label 0)
    #     explanation_moral = run_moral_explanation(trained_model, trained_tokenizer, example_context_moral, 0)
    #     print(f"**Input (Label 0):** {example_context_moral[:50]}...")
    #     print(f"**Explanation:** {explanation_moral}")

    #     # Call the function for an Immoral action (Label 1)
    #     explanation_immoral = run_moral_explanation(trained_model, trained_tokenizer, example_context_immoral, 1)
    #     print(f"\n**Input (Label 1):** {example_context_immoral[:50]}...")
    #     print(f"**Explanation:** {explanation_immoral}")

    
    # except Exception as e:
    #     print(f"An error occurred during inference/loading: {e}")