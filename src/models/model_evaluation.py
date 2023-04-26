import os

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score


def load_data(file_paths):
    X_train = pd.read_csv(file_paths['X_train'])
    X_test = pd.read_csv(file_paths['X_test'])
    y_train = pd.read_csv(file_paths['y_train'], squeeze=True)
    y_test = pd.read_csv(file_paths['y_test'], squeeze=True)
    return X_train, X_test, y_train, y_test


def load_model(file_path):
    # Load the trained model from disk
    # ... your code here ...
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))

    print("ROC-AUC Score:")
    print(roc_auc_score(y_test, y_pred))


if __name__ == '__main__':
    data_file_paths = {
        'X_train': os.path.join('..', '..', 'data', 'processed', 'X_train.csv'),
        'X_test': os.path.join('..', '..', 'data', 'processed', 'X_test.csv'),
        'y_train': os.path.join('..', '..', 'data', 'processed', 'y_train.csv'),
        'y_test': os.path.join('..', '..', 'data', 'processed', 'y_test.csv')
    }

    model_file_path = os.path.join('..', '..', 'models',
                                   'model.pkl')

    X_train, X_test, y_train, y_test = load_data(data_file_paths)
    model = load_model(model_file_path)
    evaluate_model(model, X_test, y_test)
