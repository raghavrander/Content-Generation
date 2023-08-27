import tensorflow as tf

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

SEED=34

tf.random.set_seed(SEED)

  
def generate(content_input,MAX_LEN):
    model_path = "saved_model"
    GPT2 = TFGPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    input_ids = tokenizer.encode(content_input, return_tensors='tf')

    # content_output = GPT2.generate(input_ids,do_sample = True, max_length = MAX_LEN,top_k = 50,top_p = 0.85)
                                
    generation_config = {
        "max_length": MAX_LEN,   # Maximum length of generated text
        # "min_length": 20,    # Minimum length of generated text
        "do_sample": True,
        "top_k": 0,
        "top_p": 0.8,
        "no_repeat_ngram_size": 2,
        "early_stopping":True,
    }
    content_output = GPT2.generate(input_ids, **generation_config)

    # for sample_output in enumerate(content_output):
    generated_texts = []  # Create an empty list to store generated texts

    for sample_output in content_output:
        generated_text = tokenizer.decode(sample_output, skip_special_tokens=True)
        generated_texts.append(generated_text)  # Add each generated text to the list

# Now you can return the list of generated texts or process them further
    return generated_texts
