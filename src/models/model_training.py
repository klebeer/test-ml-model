import os

import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def extract_features(data):
    y = data['is_deceptive']
    X = data.drop(columns=['is_deceptive'])
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def save_model(model, file_path):
    dump(model, file_path)


if __name__ == '__main__':
    input_path = os.path.join('..', '..', 'data', 'processed', 'processed_data.csv')
    data = load_data(input_path)

    X, y = extract_features(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model to disk
    model_file_path = os.path.join('..', '..', 'models',
                                   'model.pkl')
    save_model(model, model_file_path)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
