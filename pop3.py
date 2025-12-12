from transformers import BertTokenizer

tok = BertTokenizer.from_pretrained("bert-base-uncased")
tok.save_pretrained("./bert_moral_classifier")
