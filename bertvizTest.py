from transformers import AutoTokenizer, AutoModel, utils
from bertviz import head_view

utils.logging.set_verbosity_error()  # Suppress standard warnings
tokenizer = AutoTokenizer.from_pretrained("./bert_moral_components")
model = AutoModel.from_pretrained("bert_moral_classifier", output_attentions=True)

inputs = tokenizer.encode("I lied to someone to save my friends life", return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

html_head_view = head_view(attention, tokens, html_action='return')

with open("test.html", 'w') as file:
    file.write(html_head_view.data)
