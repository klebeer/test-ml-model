import os

from src.utils.helper_functions import clean_text, load_csv_data, save_csv_data


def preprocess_data(data):
    for i in range(10):
        column_name = f'essay{i}'
        data[column_name] = data[column_name].apply(clean_text)

    return data


if __name__ == '__main__':
    input_path = os.path.join('..', '..', 'data', 'raw', 'raw.csv')
    output_path = os.path.join('..', '..', 'data', 'processed', 'processed_data.csv')

    data = load_csv_data(input_path)
    data = preprocess_data(data)
    save_csv_data(data, output_path)
