import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    return data

def preprocess_data(data):
    # Suppression des colonnes inutiles
    data.drop(columns=['FullName', 'NationalId', 'DateOfBirth'], inplace=True)
    # Remplace les valeurs manquantes par 0
    data.fillna(0, inplace=True)
    # Encoder les variables catégorielles
    cat_columns = data.select_dtypes(include=['object']).columns.tolist()
    label_encoder = LabelEncoder()
    for col in cat_columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data

def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std

def train_models(X_train, y_train):
    models = [
        ("Logistic Regression", LogisticRegression(random_state=42, C=1.5, penalty='l2')),
        ("XGBoost Classifier", XGBClassifier(random_state=42, n_estimators=1000, max_depth=6, learning_rate=0.1, gamma=0.1, reg_alpha=0.1, reg_lambda=0.1))
    ]
    trained_models = []
    for clf_name, clf in models:
        clf.fit(X_train, y_train)
        trained_models.append((clf_name, clf))
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    results = []
    for clf_name, clf in trained_models:
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        cm = confusion_matrix(y_test, predictions)
        results.append({"Model": clf_name, "Accuracy Score": accuracy, "Roc_Auc_score": roc_auc, "Confusion Matrix": cm})
    return pd.DataFrame(results)

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(12, 6))
    for clf_name, clf in models:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=clf_name)
        roc_display.plot()
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Courbes ROC des modèles évalués')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_precision_recall_curves(models, X_test, y_test):
    plt.figure(figsize=(12, 6))
    for clf_name, clf in models:
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=clf_name)
        pr_display.plot()
    plt.title('Courbes de précision-recall des modèles évalués')
    plt.xlabel('Rappel (Recall)')
    plt.ylabel('Précision')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()

def plot_confusion_matrices(models, X_test, y_test):
    plt.figure(figsize=(12, 6))
    for clf_name, clf in models:
        predictions = clf.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.title(f'Matrice de confusion pour le modèle {clf_name}')
        plt.grid(False)
        plt.show()

def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_proba_individual(model, data):
    proba = model.predict_proba(data)[:, 1]
    return proba

def main():
    # Chargement des données
    data = load_data('donnee_bic_matar_cleaned_final.csv')
    # Prétraitement des données
    data_processed = preprocess_data(data)
    # Division données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = split_data(data_processed, 'Target')
    # Standardisation des données
    X_train_std, X_test_std = standardize_data(X_train, X_test)
    # Entraînement des modèles
    trained_models = train_models(X_train_std, y_train)
    # Évaluation des modèles
    results = evaluate_models(trained_models, X_test_std, y_test)
    print(results)
    # Tracer les courbes ROC, de précision-rappel et les matrices de confusion
    plot_roc_curves(trained_models, X_test_std, y_test)
    plot_precision_recall_curves(trained_models, X_test_std, y_test)
    plot_confusion_matrices(trained_models, X_test_std, y_test)
    # Sauvegarde du modèle XGBoost 
    xgb_model = [model for model in trained_models if model[0] == 'XGBoost Classifier'][0][1]
    save_model(xgb_model, 'xgboost_model.pkl')
    # Charger le modèle sauvegardé
    loaded_model = load_model('xgboost_model.pkl')
    individual_data = X_test.iloc[:3] 
    for i, row in individual_data.iterrows():
        proba = predict_proba_individual(loaded_model, row.to_numpy().reshape(1, -1))
        print(f"Individu {i} - Probabilité de défaut : {proba}")

if __name__ == "__main__":
    main()
