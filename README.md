Citations:

for github Help asked chatGPT:

- Can you help me with this github error: "Enumerating objects: 34838, done. Counting objects: 100% (34838/34838), done. Delta compression using up to 24 threads Compressing objects: 100% (24972/24972), done. error: RPC failed; HTTP 500 curl 22 The requested URL returned error: 500 send-pack: unexpected disconnect while reading sideband packet Writing objects: 100% (34838/34838), 5.14 GiB | 117.38 MiB/s, done. Total 34838 (delta 9608), reused 34838 (delta 9608), pack-reused 0 fatal: the remote end hung up unexpectedly Everything up-to-date"

For help with github I used these two stack overflow pages:
https://stackoverflow.com/questions/63727594/github-git-checkout-returns-error-invalid-path-on-windows
https://stackoverflow.com/questions/11941175/git-fetch-pull-clone-hangs-on-receiving-objects

For general information on AI models to help assist knowing when to use which one: -https://medium.com/@reyhaneh.esmailbeigi/bert-gpt-and-bart-a-short-comparison-5d6a57175fca

While attempting to use a BART model to summarize components from a CSV I aksed chatGPT : "This code just outputs the input, why is that happening? "from transformers import BartForConditionalGeneration, BartTokenizer model_name = "facebook/bart-large" tokenizer = BartTokenizer.from_pretrained(model_name) model = BartForConditionalGeneration.from_pretrained(model_name) prompt = ( "Norm: It's kind to help your parent " + "Situation: Peter is at home when he gets a phone call from her mother asking him to come over and help her move a couch." + "Intention: Peter wants to respond to his mother's request."+ "Moral Action: Peter tells his mother that he can and says that he'll be over at her place shortly."+ "Moral Consequence: Peter's mother is delighted and relieved that she will get the assistance she needed."+ "Please combine these parts to create a single explanation for this events: " ) inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512) # print("INPUT TOKENS:", len(inputs["input_ids"][0])) summary_ids = model.generate( inputs["input_ids"], num_beams=4, max_length=60, min_length=12, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True ) explanation = tokenizer.decode(summary_ids[0], skip_special_tokens=True) print(explanation)""
This was very helpful as it let me get a proper output from my test, but also let me know that a BART model might not be the best for this task.
