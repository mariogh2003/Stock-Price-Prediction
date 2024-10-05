import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from dataset_manipulation import pre_processing, final_csv
import seaborn as sns
import numpy as np

# Define file paths
train_file_path = "C:/Users/mario/Downloads/training_dataset.csv"
test_file_path = "C:/Users/mario/Downloads/testing_dataset.csv"
combined_news_path = "C:/Users/mario/Downloads/Combined_News_DJIA.csv"
embedding_csv = "C:/Users/mario/Downloads/Word_embedding.csv"
merged_csv = "C:/Users/mario/Downloads/Merged_dataset.csv"

def load_data(train_file_path, test_file_path):
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    return train, test

def preprocess_data(train, test):
    data = pd.concat([train, test])
    data = pre_processing(data)
    return data

def split_data(data, train_num):
    train = data[:train_num]
    test = data[train_num:]
    return train, test

def prepare_features_and_labels(train, test):
    y_train = train["pct_change"].values
    X_train = train.drop("pct_change", axis=1).values.astype(float)
    y_test = test["pct_change"].values
    X_test = test.drop("pct_change", axis=1).values.astype(float)
    return X_train, y_train, X_test, y_test

def impute_and_normalize(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def classify_pct_change(y, bins, labels):
    return pd.cut(y, bins=bins, labels=labels).fillna(0)

def train_model(X_train, y_train):
    mlp = MLPClassifier(max_iter=5000)
    param_grid = {
        'hidden_layer_sizes': [(16)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    reg_cv = GridSearchCV(mlp, param_grid, verbose=1, n_jobs=-1)
    reg_cv.fit(X_train, y_train)
    best_params = reg_cv.best_params_

    mlp = MLPClassifier(**best_params, max_iter=5000)
    mlp.fit(X_train, y_train)

    return mlp

def evaluate_model(mlp, X_train, y_train_classified, X_test, y_test_classified, labels):
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)

    print("Distribution of predictions:")
    print(pd.Series(train_predictions).value_counts())

    train_score = mlp.score(X_train, y_train_classified)
    test_score = mlp.score(X_test, y_test_classified)
    conf_matrix = confusion_matrix(y_test_classified, test_predictions, labels=range(2))

    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_report(y_test_classified, test_predictions)}")

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return test_predictions

def training_main():
    final_csv()
    train, test = load_data(train_file_path, test_file_path)
    train_num = train.shape[0]

    data = preprocess_data(train, test)
    train, test = split_data(data, train_num)

    X_train, y_train, X_test, y_test = prepare_features_and_labels(train, test)
    X_train, X_test = impute_and_normalize(X_train, X_test)

    bins = [-np.inf, 0, np.inf]
    labels = [1, 0]

    y_train_classified = classify_pct_change(y_train, bins, labels).astype(int)
    y_test_classified = classify_pct_change(y_test, bins, labels).astype(int)

    mlp = train_model(X_train, y_train_classified)
    test_predictions = evaluate_model(mlp, X_train, y_train_classified, X_test, y_test_classified, labels)

    return test_predictions

if __name__ == "__main__":
    training_main()
