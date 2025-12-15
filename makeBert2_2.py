# To generate this code I asked chatGPT: "Can you create me a python file which fine tunes a bert machine learning model. The model should output either 0 or 1, which corresponds with a moral decision of moral or immoral. The model should be fine tuned with two csv files, the first is "./testingSets/dataSet_1_train.csv" and the second is "./data_With_Sentence copy 2.csv". For the first csv include any row in which "is_short" is True, then the input should be what is the "input" collum and that is matched with the label. For the second csv the input should be the "explanation" collumn which is matched with the boolean opposite of the "label" collumn. This new model should be able to be called from a function, which returns either the words "MORAL" if the ouput is 0 or "IMMORAL" if the output is 1. "
# I then asked "What are the diffrent bert models I could pretrain ontop of, and what are their bennifits?" to make sure it chose a good base model

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./moral_bert_model_short_sentences"

############################################
# Dataset
############################################

class MoralDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


############################################
# Load & Prepare Data
############################################

def load_training_data():
    texts = []
    labels = []

    # -------- Dataset 1 --------
    df1 = pd.read_csv("./testingSets/dataSet_1_train.csv")

    df1 = df1[df1["is_short"] == "True"]

    for _, row in df1.iterrows():
        texts.append(row["input"])
        labels.append(int(row["label"]))  # assumed already 0 or 1

    # -------- Dataset 2 --------
    df2 = pd.read_csv("./data_With_Sentence copy 2.csv")

    for _, row in df2.iterrows():
        texts.append(row["explanation"])
        labels.append(1 - int(row["label"]))  # boolean opposite

    return texts, labels


############################################
# Training
############################################

def train_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    texts, labels = load_training_data()

    dataset = MoralDataset(texts, labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Model training complete and saved.")


############################################
# Inference Function
############################################

_tokenizer = None
_model = None

def predict_moral_status(text: str) -> str:
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        _tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR)
        _model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR)
        _model.eval()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "MORAL" if prediction == 0 else "IMMORAL"


############################################
# Entry Point
############################################

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0))
    train_model()
