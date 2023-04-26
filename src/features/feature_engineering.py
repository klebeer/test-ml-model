import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def extract_features(data):
    y = data['is_deceptive']
    X = data.drop(columns=['is_deceptive'])
    return X, y


def preprocess_features(X):
    # Preprocess numerical features
    numerical_features = ['age', 'height', 'income']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Preprocess categorical features
    categorical_features = ['status', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education',
                            'ethnicity', 'job', 'offspring', 'pets', 'religion', 'sign', 'smokes', 'speaks']
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_categories = encoder.fit_transform(X[categorical_features])
    encoded_categories_df = pd.DataFrame(encoded_categories.toarray(),
                                         columns=encoder.get_feature_names_out(categorical_features))

    # Merge the preprocessed numerical and categorical features
    X_preprocessed = X.drop(columns=categorical_features)
    X_preprocessed = pd.concat([X_preprocessed, encoded_categories_df], axis=1)

    # Preprocess text features
    text_features = ['essay0', 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7', 'essay8', 'essay9']
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_essays = vectorizer.fit_transform(
        X['essay0'].values.astype('U'))
    X_essays_df = pd.DataFrame(X_essays.toarray(), columns=vectorizer.get_feature_names_out())

    # Merge the preprocessed text features with the other features
    X_preprocessed = pd.concat([X_preprocessed, X_essays_df], axis=1)

    return X_preprocessed
