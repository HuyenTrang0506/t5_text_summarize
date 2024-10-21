#model_preparation.py
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Prepare Hugging Face dataset
def prepare_dataset(df):
    dataset = Dataset.from_pandas(df)
    return dataset

# Preprocessing function for tokenization
def preprocess_function(examples, tokenizer):
    inputs = ["summarize: " + ' '.join(comment.split('.')) for comment in examples['comment']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(examples['comment tom tat'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    # Replace padding token id with -100
    model_inputs["labels"] = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in model_inputs["labels"]]
    
    return model_inputs

# Tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

# Load tokenizer and model
def load_model_tokenizer(model_name='t5-small'):
    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer
