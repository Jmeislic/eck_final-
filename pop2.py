from transformers import BertTokenizerFast, BertForSequenceClassification

path = "./bert_moral_classifier"

print("Loading tokenizer...")
tok = BertTokenizerFast.from_pretrained(path)
print("Tokenizer OK")

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(
    path,
    use_safetensors=True
)
print("Model OK")
