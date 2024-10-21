from data_processing import load_data, clean_data, save_cleaned_data
from model_preparation import load_model_tokenizer, prepare_dataset, tokenize_dataset
from model_training import train_model, save_tokenizer
from evaluation import create_summarizer, generate_summaries, evaluate_model
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Abbreviation dictionary
abbreviation_dict = {
    'ko': 'không',
    'ip': 'iPhone',
    'bt': 'biết',
    'prm': 'promax',
    'vs': 'với',
    'm': 'mình',
    'ok': 'ổn',
    'quay vd': 'quay video',
    'cóc sạc': 'cốc sạc',
    'hư': 'hỏng',
    'copy': 'sao chép',
    'cam': 'camera',
    'bx': 'bữa',
    'app': 'ứng dụng',
    'lỏ': 'không tốt',
    'mh': 'màn hình',
    'seeder': 'nguồn tin tức không uy tín',
    'éo': 'không',
    'bonus': 'thêm',
    'ngót': 'gần',
    'tr': 'triệu',
    'e': 'em',
    'răt': 'rất',
    'and': 'android',
    'auto': 'tự động',
    'flagship': 'sản phẩm tiên phong',
    'sd': 'sử dụng',
    'andr': 'android',
    # Thêm các phiên bản iPhone vào xử lý
    'ip6': 'iPhone 6',
    'ip7': 'iPhone 7',
    'ip8': 'iPhone 8',
    'ip9': 'iPhone 9',
    'ip10': 'iPhone 10',
    'ip11': 'iPhone 11',
    'ip12': 'iPhone 12',
    'ip13': 'iPhone 13',
    'ip14': 'iPhone 14',
    'ip15': 'iPhone 15',
    'ipx': 'iPhone X',
}

if __name__ == "__main__":
    # Step 1: Load and clean data
    file_path = 'crawl_data.xlsx'
    df = load_data(file_path)
    df_cleaned = clean_data(df, abbreviation_dict)
    save_cleaned_data(df_cleaned, 'cleaned_crawl_data.xlsx')

    # Step 2: Prepare dataset
    dataset = prepare_dataset(df_cleaned)

    # Split dataset into train and test sets (80% train, 20% test)
    train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)

    # Convert to Hugging Face dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Load tokenizer and model
    model, tokenizer = load_model_tokenizer()

    # Tokenize both train and test datasets
    tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer)
    tokenized_test_dataset = tokenize_dataset(test_dataset, tokenizer)

    # Combine tokenized datasets into one dictionary
    tokenized_dataset = {
        'train': tokenized_train_dataset,
        'test': tokenized_test_dataset
    }

    # Step 3: Train the model
    trained_model = train_model(model, tokenized_dataset)
    save_tokenizer(tokenizer)

    # Step 4: Evaluate the model
    summarizer = create_summarizer()
    predicted_summaries = generate_summaries(tokenized_dataset, summarizer)

    # Get actual summaries from the test dataset
    actual_summaries = [example['comment tom tat'] for example in test_df]

    # Step 5: Calculate evaluation metrics
    rouge_results, f1 = evaluate_model(predicted_summaries, actual_summaries)

    # Print the results
    print("ROUGE Results:", rouge_results)
    print("F1 Score:", f1)
