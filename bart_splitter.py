import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration


#To start this code base I asked ChatGPT: "Can you create for me a python program which creates an instance of the AI model BART, which creates a sentence describing a situation from a csv. The csv contains 8 collums, "norm" "situation" "intention" "moral_action" "moral_consequence" "immoral_action" "immoral_consequence" and "label". The output from this model should an explanation for why an action is either moral or immoral depending on the label, where a moral action is a label of 0 and an immoral action is a label of 1."

# -------------------------------------------------------
# Load BART model and tokenizer
# -------------------------------------------------------
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# -------------------------------------------------------
# Function to create a description prompt from a CSV row
# -------------------------------------------------------
def build_prompt(row):
    """
    Build a natural-language prompt describing the situation,
    and asking the model to explain why the action is moral or immoral.
    """

    moral_label = "moral" if row["label"] == 0 else "immoral"

    prompt = (
        f"Norm: {row['norm']}\n"
        f"Situation: {row['situation']}\n"
        f"Intention: {row['intention']}\n"
        f"Moral Action: {row['moral_action']}\n"
        f"Moral Consequence: {row['moral_consequence']}\n"
        f"Immoral Action: {row['immoral_action']}\n"
        f"Immoral Consequence: {row['immoral_consequence']}\n\n"
        f"Given the situation above, the action is considered {moral_label}. "
        f"Explain why the action is {moral_label}."
    )

    return prompt


# -------------------------------------------------------
# Function to generate explanation using BART
# -------------------------------------------------------
def generate_explanation(prompt, max_len=128):
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# -------------------------------------------------------
# Load CSV and generate explanations
# -------------------------------------------------------
def run(csv_path, output_path="model_output.csv"):
    df = pd.read_csv(csv_path)

    outputs = []

    for i, row in df.iterrows():
        prompt = build_prompt(row)
        explanation = generate_explanation(prompt)
        outputs.append(explanation)
        print(f"\nEntry {i} → Explanation:\n{explanation}\n")

    df["bart_explanation"] = outputs
    df.to_csv(output_path, index=False)
    print(f"\nAll explanations saved to {output_path}")


# -------------------------------------------------------
# Main call
# -------------------------------------------------------
if __name__ == "__main__":
    run("moral_stories_csv/data_classification_action+context+consequence_lexical_bias_train.csv")   # change to your filename
