#To make this file I asked ChatGPT : "Can you make a python function which creates a callable BERT model. The BERT should be finetuned on a dataset called "data_With_Sentence.csv". The model should take as an input a sentence and output different moral components. Each row of the dataset contains "ID,norm,situation,intention,moral_action,moral_consequence,label,immoral_action,immoral_consequence,explanation", ID should be ignored, and the training item should be explanation, without outputs of norm, situation,intention, moral_action, moral_consequence, label,immoral_action, immoral_consequence. The csv does not always have data for all of the rows, if that happens still train insert the empty string. The file should have a line that makes the model, as well as a function that lets the model be called from other files."
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------
class MoralDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        df = pd.read_csv(csv_path)

        # Ensure all relevant columns exist, fill missing with ""
        cols = [
            "explanation", "norm", "situation", "intention",
            "moral_action", "moral_consequence",
            "immoral_action", "immoral_consequence"
        ]
        for c in cols:
            if c not in df:
                df[c] = ""
            df[c] = df[c].fillna("")

        self.texts = df["explanation"].tolist()

        # Targets (one label per component)
        self.targets = {
            "norm": df["norm"].tolist(),
            "situation": df["situation"].tolist(),
            "intention": df["intention"].tolist(),
            "moral_action": df["moral_action"].tolist(),
            "moral_consequence": df["moral_consequence"].tolist(),
            "immoral_action": df["immoral_action"].tolist(),
            "immoral_consequence": df["immoral_consequence"].tolist(),
        }

        # Build label lookup tables (multi-task classification)
        self.label_maps = {
            k: {lbl: i for i, lbl in enumerate(sorted(list(set(v))))}
            for k, v in self.targets.items()
        }

        # Convert labels to IDs
        for k in self.targets:
            self.targets[k] = [
                self.label_maps[k][lbl] for lbl in self.targets[k]
            ]

        self.tokenizer = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = {k: self.targets[k][idx] for k in self.targets}
        return item


# -----------------------------------------------------------
# BERT Multi-Task Model
# -----------------------------------------------------------
class MoralBERT(nn.Module):
    def __init__(self, num_labels_dict):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # classification head for each moral component
        self.heads = nn.ModuleDict({
            name: nn.Linear(self.bert.config.hidden_size, n_labels)
            for name, n_labels in num_labels_dict.items()
        })

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output

        logits = {name: head(pooled) for name, head in self.heads.items()}

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = sum(loss_fn(logits[name], labels[name]) for name in logits)

        return logits, loss


# -----------------------------------------------------------
# Training function
# -----------------------------------------------------------
def train_moral_bert(csv_path="data_With_Sentence.csv", save_path="./moral_bert"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    dataset = MoralDataset(csv_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MoralBERT(num_labels_dict={
        k: len(dataset.label_maps[k])
        for k in dataset.label_maps
    })

#This forum post let me fix this error https://github.com/huggingface/transformers/issues/36954
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            input_ids = batch["input_ids"]
            mask = batch["attention_mask"]
            labels = batch["labels"]

            logits, loss = model(
                input_ids=input_ids,
                attention_mask=mask,
                labels=labels
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} completed.")

    # Save model + tokenizer + label maps
    torch.save(model.state_dict(), f"{save_path}/model.pt")
    tokenizer.save_pretrained(save_path)
    torch.save(dataset.label_maps, f"{save_path}/label_maps.pt")

    print("Training complete. Model saved.")

    # Return callable classifier
    return load_moral_bert(save_path)


# -----------------------------------------------------------
# Loading function (for use in other files)
# -----------------------------------------------------------
def load_moral_bert(model_path="./moral_bert"):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    label_maps = torch.load(f"{model_path}/label_maps.pt")

    model = MoralBERT(num_labels_dict={
        k: len(label_maps[k])
        for k in label_maps
    })
    model.load_state_dict(torch.load(f"{model_path}/model.pt"))
    model.eval()

    # Reverse label maps
    id2label = {k: {v: k2 for k2, v in labels.items()}
                for k, labels in label_maps.items()}

    # Callable function
    def classifier(sentence):
        enc = tokenizer(sentence, return_tensors="pt", truncation=True)

        logits, _ = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )

        output = {}
        for comp in logits:
            pred_id = torch.argmax(logits[comp], dim=-1).item()
            output[comp] = id2label[comp][pred_id]

        return output

    return classifier, model, tokenizer

train_moral_bert()