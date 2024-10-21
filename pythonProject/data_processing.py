#data_processing.py
import pandas as pd
import re

# Load dataset
def load_data(file_path, sheet_name='Summarize'):
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Print the columns of the dataset
    print("Columns in the dataset:", df.columns.tolist())
    return df

# Replace abbreviations
def replace_abbreviations(text, abbr_dict):
    for abbr, full_form in abbr_dict.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_form, text)
    return text

# Remove special characters
def remove_special_characters(text):
    return re.sub(r'[^A-Za-zÀ-ỹ0-9\s]', '', text)

# Clean the data
def clean_data(df, abbreviation_dict):
    # Drop NaN values
    df_cleaned = df.dropna(subset=['comment', 'comment tom tat'])

    # Replace abbreviations and remove special characters
    df_cleaned['comment'] = df_cleaned['comment'].apply(lambda x: replace_abbreviations(x, abbreviation_dict))
    df_cleaned['comment'] = df_cleaned['comment'].apply(remove_special_characters)

    df_cleaned['comment tom tat'] = df_cleaned['comment tom tat'].apply(lambda x: replace_abbreviations(x, abbreviation_dict))
    df_cleaned['comment tom tat'] = df_cleaned['comment tom tat'].apply(remove_special_characters)

    # Remove rows with empty comments or summaries
    df_cleaned = df_cleaned[df_cleaned['comment'].str.strip() != ""]
    df_cleaned = df_cleaned[df_cleaned['comment tom tat'].str.strip() != ""]

    return df_cleaned

def save_cleaned_data(df_cleaned, output_file, sheet_name='Summarize'):
    with pd.ExcelWriter(output_file) as writer:
        df_cleaned.to_excel(writer, sheet_name=sheet_name, index=False)