import os

from src.data.data_preparation import preprocess_data
from src.features.feature_engineering import load_data, extract_features, preprocess_features
from src.models.model_evaluation import load_model, evaluate_model
from src.models.model_training import train_model, save_model
from utils.helper_functions import load_csv_data, save_csv_data


def main():
    # Data preparation
    raw_data_path = os.path.join('..', 'data', 'raw', 'raw.csv')
    raw_data = load_csv_data(raw_data_path)
    preprocessed_data = preprocess_data(raw_data)
    preprocessed_data_path = os.path.join('..', 'data', 'processed', 'processed_data.csv')
    save_csv_data(preprocessed_data, preprocessed_data_path)

    # Feature engineering
    data = load_data(preprocessed_data_path)
    X, y = extract_features(data)
    X_preprocessed = preprocess_features(X)

    # Train and save the model
    model = train_model(X_preprocessed, y)
    model_file_path = os.path.join('..', 'models',
                                   'model.pkl')
    save_model(model, model_file_path)

    # Evaluate the model
    model = load_model(model_file_path)
    evaluate_model(model, X_preprocessed, y)


if __name__ == '__main__':
    main()
