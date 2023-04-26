import os
import re

import pandas as pd


def clean_text(text):
    if not isinstance(text, str):
        return ""

    cleaned_text = re.sub('<[^>]*>', '', text)
    cleaned_text = re.sub('[^a-zA-Z\s]', '', cleaned_text)
    cleaned_text = cleaned_text.lower().strip()
    return cleaned_text


def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data


def save_csv_data(data, file_path):
    data.to_csv(file_path, index=False)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def write_text_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
