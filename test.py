from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

prompt = (
        "summarize: " +
        "Norm: It's kind to help your parent " + 
        "Situation: Peter is at home when he gets a phone call from her mother asking him to come over and help her move a couch." +
        "Intention: Peter wants to respond to his mother's request."+
        "Moral Action: Peter tells his mother that he can and says that he'll be over at her place shortly."+
        "Moral Consequence: Peter's mother is delighted and relieved that she will get the assistance she needed."
       
        
    )

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
# print("INPUT TOKENS:", len(inputs["input_ids"][0]))

summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=80,
    min_length=12,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True
)

explanation = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(explanation)