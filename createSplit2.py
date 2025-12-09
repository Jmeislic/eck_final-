#this is a slightly changed copy of createSplitBert.py
# To generate this code I asked Google Gemini: Can you create for me a python program which trains a custom instance of the AI model BART, which from splits an inputted sentence into "norm", "situation", "intention", "moral_action" or "immoral_action" (depending on the label, 0 for unethical and 1 for ethical). The training dataset is called cleaned_data.csv and the training input is the collum explanation, and the training output is "norm", "situation", "intention", "moral_action" or "immoral_action" (depending on the label, 0 for unethical and 1 for ethical). Please create a file which creates this finetuned model, and has a function which allows it to be called. As well in training this model only include moral action in the output if label is 0, and only include immoral action in the output if the label is 1.
#Then I aksed the follow up question: With this function it seems that it will add in columns I would not want like ID, which are in the csv file
# I then asked it "could you make the code above threaded so it goes faster?", and "could you make the code above threaded so it goes faster?" to help with using pytorch to make things go faster

import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
# --- Configuration and Placeholders ---
MODEL_NAME = "facebook/bart-base"
MODEL_OUTPUT_DIR = "./bertParsed2"
TRAINING_DATASET = "extra_cleaned_data.csv"

# **PLACEHOLDERS:** Replace these with your actual column names in a non-sensitive context.
INPUT_COLUMN = "explanation"
STATIC_OUTPUT_COLUMNS = [ "norm", "situation", "intention", "action"] # Equivalent to norm, situation, intention
LABEL_COLUMN = "label"                                               # Column containing 0 or 1
# CONDITION_0_COLUMN = "immoral_action"                               # Equivalent to immoral_action (Label=0)
# CONDITION_1_COLUMN = "moral_action"                               # Equivalent to moral_action (Label=1)

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# --- Setup ---
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# --- 1. Data Transformation (Incorporates Conditional Logic) ---
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the DataFrame for Seq2Seq training, creating the target text 
    conditionally based on the 'label' column's value.
    """
    
    def generate_target_text(row):
        target_parts = []
        
        # 1. Include the static components (norm, situation, intention)
        for col in STATIC_OUTPUT_COLUMNS:
            content = str(row.get(col, "")).strip()
            target_parts.append(f"{col}: {content}")
            
        # 2. Include the conditional component (moral_action OR immoral_action)
        label_value = row.get(LABEL_COLUMN)
        
        # if label_value == 0:
        #     # If label is 0 (unethical), include the 'immoral_action' equivalent
        #     col = CONDITION_0_COLUMN
        #     content = str(row.get(col, "")).strip()
        #     target_parts.append(f"{col}: {content}")
        # elif label_value == 1:
        #     # If label is 1 (ethical), include the 'moral_action' equivalent
        #     col = CONDITION_1_COLUMN
        #     content = str(row.get(col, "")).strip()
        #     target_parts.append(f"{col}: {content}")
        
        # Join all parts to form the final structured output string
        return " ".join(target_parts)

    df['target_text'] = df.apply(generate_target_text, axis=1)
    
    # Select ONLY the required columns for the tokenizer (input and generated target)
    df = df[[INPUT_COLUMN, 'target_text']]
    return df

# --- 2. Tokenization ---
def tokenize_function(examples):
    """Tokenizes the input and target texts for the model."""
    # Tokenize the input text
    model_inputs = tokenizer(
        examples[INPUT_COLUMN], 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize the target text
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['target_text'], 
            max_length=MAX_TARGET_LENGTH, 
            truncation=True,
            padding="max_length"
        )
    
    # Replace padding tokens with -100 so they are ignored by the loss function
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- 3. Training Function ---
def train_decomposition_model():
    """Reads data, preprocesses, and fine-tunes the BART model."""
    print("🚀 Starting BART fine-tuning process...")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    try:
        # Load and prepare the data
        df = pd.read_csv(TRAINING_DATASET)
        df_processed = prepare_data(df)
        
        # Convert pandas DataFrame to Hugging Face Dataset object
        dataset = Dataset.from_pandas(df_processed)
        
        # Split into training and validation sets
        train_test_split = dataset.train_test_split(test_size=0.1)
        tokenized_datasets = train_test_split.map(tokenize_function, batched=True)

        # Training Arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            num_train_epochs=3,                     
            per_device_train_batch_size=8,         
            per_device_eval_batch_size=8,           
            logging_steps=100,
            save_strategy="epoch",                  
            eval_strategy="epoch",            
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            predict_with_generate=True,     
            fp16=True,         
        )

        # Data Collator 
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Initialize and run the Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        
        # Save the fine-tuned model and tokenizer
        model.save_pretrained(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        print(f"✅ Training complete. Model saved to {MODEL_OUTPUT_DIR}")

    except FileNotFoundError:
        print(f"Error: The training file '{TRAINING_DATASET}' was not found. Please ensure it is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")

# --- 4. Inference Function ---
def generate_decomposition(input_sentence: str) -> str:
    """Loads the fine-tuned model and generates the decomposed output."""
    try:
        # Load the fine-tuned model and tokenizer
        loaded_tokenizer = BartTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
        loaded_model = BartForConditionalGeneration.from_pretrained(MODEL_OUTPUT_DIR)

        # Tokenize the input sentence
        inputs = loaded_tokenizer(
            [input_sentence], 
            max_length=MAX_INPUT_LENGTH, 
            return_tensors="pt", 
            truncation=True
        )

        # Generate the output sequence
        output_ids = loaded_model.generate(
            inputs["input_ids"], 
            max_length=MAX_TARGET_LENGTH, 
            num_beams=4, # Use beam search for higher quality
            early_stopping=True
        )

        # Decode and return the generated text
        output_text = loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    except Exception as e:
        return f"Error loading or running the model: {e}. Ensure the training function was run successfully first."

# --- Main Execution Block ---
if __name__ == '__main__':
    # To run the program, you would first call the training function:
    # train_decomposition_model() 

    # After training, you can call the inference function:
    # test_sentence = "Oliver North, a retired Marine Lieutenant Colonel, became famous for his central role in the 1980s Iran-Contra affair, a scandal where he helped facilitate secret arms sales to Iran (despite a U.S. ban) and funneled the profits to fund the Contra rebels in Nicaragua, which Congress had prohibited. He admitted to lying to Congress and shredding documents but gained public notoriety during his televised testimony, though his subsequent convictions were overturned due to his immunized testimony. "
    while True:
        test_sentence = input("Ask Jesses model")
        decomposition = generate_decomposition(test_sentence)
        print(f"\nInput: {test_sentence}")
        print(f"Output: {decomposition}")
    # pass