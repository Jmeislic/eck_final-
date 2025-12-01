import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# -------------------------------
# GPU CHECK
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA:", torch.cuda.get_device_name(0))

# -------------------------------
# LOAD CSV
# -------------------------------
df = pd.read_csv("moral_stories_csv/data_moral_stories_full.csv")

# -------------------------------
# PREPROCESSING
# -------------------------------
# Prepare moral (ethical) actions
moral_df = df[['moral_action', 'norm', 'situation', 'intention', 'moral_consequence']].copy()
moral_df.rename(columns={'moral_action': 'action', 'moral_consequence': 'consequence'}, inplace=True)
moral_df['label'] = 1  # ethical

# Prepare immoral (unethical) actions
immoral_df = df[['immoral_action', 'norm', 'situation', 'intention', 'immoral_consequence']].copy()
immoral_df.rename(columns={'immoral_action': 'action', 'immoral_consequence': 'consequence'}, inplace=True)
immoral_df['label'] = 0  # unethical

# Combine and drop missing actions
combined_df = pd.concat([moral_df, immoral_df], ignore_index=True)
combined_df.dropna(subset=['action'], inplace=True)

# Combine all reasoning/context columns into one text input
combined_df['text'] = combined_df[['norm', 'situation', 'intention', 'action', 'consequence']].agg(' '.join, axis=1)

# -------------------------------
# TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 128  # adjust if your reasoning text is long

encodings = tokenizer(
    combined_df['text'].tolist(),
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt"
)

labels = torch.tensor(combined_df['label'].values)

dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)

# -------------------------------
# TRAIN/VAL SPLIT
# -------------------------------
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# -------------------------------
# MODEL
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

epochs = 4
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1)
    return np.sum(pred_flat == labels) / len(labels)

def format_time(t):
    return str(datetime.timedelta(seconds=int(t)))

# -------------------------------
# TRAINING LOOP
# -------------------------------
training_stats = []
total_t0 = time.time()

for epoch_i in range(epochs):
    print(f"\n======== Epoch {epoch_i+1} / {epochs} ========")
    print("Training...")
    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(
            b_input_ids,
            attention_mask=b_mask,
            labels=b_labels
        )

        loss = outputs.loss
        logits = outputs.logits

        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 and step != 0:
            print(f"  Batch {step} of {len(train_dataloader)} processed.")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"  Average training loss: {avg_train_loss:.4f}")
    print(f"  Training epoch time: {format_time(time.time() - t0)}")

    # -------------------------------
    # VALIDATION
    # -------------------------------
    print("Running Validation...")
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                attention_mask=b_mask,
                labels=b_labels
            )

        loss = outputs.loss
        logits = outputs.logits
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        labels_flat = b_labels.cpu().numpy()
        total_eval_accuracy += flat_accuracy(logits, labels_flat)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    avg_val_loss = total_eval_loss / len(val_dataloader)

    print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")

    training_stats.append({
        'epoch': epoch_i+1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'val_acc': avg_val_accuracy
    })

print("\nTraining Complete!")
print(f"Total training time: {format_time(time.time() - total_t0)}")

# -------------------------------
# SAVE MODEL
# -------------------------------
output_dir = "./bert_moral_classifier/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to", output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
