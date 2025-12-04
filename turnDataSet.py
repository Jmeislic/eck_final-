## To create this codebase I asked chatGPT "Can you write me a python file which takes in a csv at file path ".\moral_stories_csv\data_classification_action+context+consequence_lexical_bias_train.csv", and goes through each row of the dataset, taking in each item in the row (minus label), and then using that information and a query BART Model constructs a sentence that turns the different information into one sentence explaining the moral situation. The program then writes the row into a new csv file called data_With_Sentence, with a new collum labeled explanation" which contains the created sentence."

import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

INPUT_CSV = ".\moral_stories_csv\data_classification_action+context+consequence_lexical_bias_train.csv"
OUTPUT_CSV = "data_With_Sentence.csv"

# Load BART model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def generate_explanation(row, label_col="ID"):
    # Build a small paragraph from all columns except label
    parts = []
    for col, val in row.items():
        if col != label_col:
            parts.append(f"{col}: {val}")
    situation_text = " ".join(parts)

    # BART-friendly summarization prompt
    prompt = (
        "Summarize this situation:"
        f"{situation_text}\n\n"
        
    )
    # print("PROMPT:", prompt)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    # print("INPUT TOKENS:", len(inputs["input_ids"][0]))

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=100,
        min_length=12,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    explanation = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return explanation



def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV)
    df = df.fillna("")
    print("Generating explanations...")
    explanations = []

    for _, row in df.iterrows():
        explanation = generate_explanation(row)
        explanations.append(explanation)
        print(row["ID"])
        # print(explanation)
        # print("")

    df["explanation"] = explanations

    print("Saving output CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Done! Created {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
