#evaluation.py
from transformers import pipeline
import evaluate
from sklearn.metrics import precision_recall_fscore_support

# Load evaluation metrics
rouge = evaluate.load("rouge")

# Create summarization pipeline
def create_summarizer():
    return pipeline("summarization", model='./t5_fine_tuned_model', tokenizer='./t5_fine_tuned_model')

# Generate summaries
def generate_summaries(tokenized_dataset, summarizer):
    predicted_summaries = []
    for example in tokenized_dataset['test']:
        input_text = "summarize: " + example['comment']
        predicted_summary = summarizer(input_text, max_length=128, min_length=30, do_sample=False)
        if predicted_summary and predicted_summary[0].get('summary_text'):
            predicted_summaries.append(predicted_summary[0]['summary_text'])
        else:
            predicted_summaries.append("Tóm tắt không khả dụng")
    return predicted_summaries

# Evaluate model using ROUGE and F1 Score
def evaluate_model(predicted_summaries, actual_summaries):
    # Filter invalid summaries
    filtered_predicted = [pred for pred in predicted_summaries if pred != "Tóm tắt không khả dụng"]
    filtered_actual = [actual for pred, actual in zip(predicted_summaries, actual_summaries) if pred != "Tóm tắt không khả dụng"]

    # Calculate ROUGE score
    rouge_results = rouge.compute(predictions=filtered_predicted, references=filtered_actual, use_stemmer=True)

    # Tokenize actual and predicted summaries
    predicted_tokens = [summary.split() for summary in filtered_predicted]
    actual_tokens = [summary.split() for summary in filtered_actual]

    # Flatten the lists for comparison
    flat_predicted = [token for sublist in predicted_tokens for token in sublist]
    flat_actual = [token for sublist in actual_tokens for token in sublist]

    # Calculate F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(flat_actual, flat_predicted, average='weighted', zero_division=0)

    # Return evaluation results
    return rouge_results, f1
