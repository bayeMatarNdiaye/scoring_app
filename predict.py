import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

def load_model(file_path):
    """
    Load the trained model from the specified file path.

    Parameters:
    file_path (str): The path to the trained model file.

    Returns:
    model: The loaded trained model.
    """
    model = joblib.load(file_path)
    return model

def preprocess_data(data):
    """
    Perform preprocessing on the dataset.

    Parameters:
    data (pandas.DataFrame): The dataset to preprocess.

    Returns:
    pandas.DataFrame: The preprocessed dataset.
    """
    data.fillna(0, inplace=True)
    label_encoder = LabelEncoder()
    for category in data.select_dtypes(include=['object']).columns.tolist():
        data[category] = label_encoder.fit_transform(data[category])
    if 'Target' in data.columns:
        data.drop(columns=['FullName', 'NationalId', 'DateOfBirth', 'Target'], inplace=True)
    else:
        data.drop(columns=['FullName', 'NationalId', 'DateOfBirth'], inplace=True)
    return data

def predict_proba_individual(model, data):
    """
    Make a probability prediction for an individual using the trained model.

    Parameters:
    model: The trained model.
    data (pandas.DataFrame): The data for the individual.

    Returns:
    float: The predicted probability of the individual belonging to the positive class.
    """
    proba = model.predict_proba(data)[:, 1]
    return proba

def assign_score(proba):
    """
    Assign a score based on the predicted probability.

    Parameters:
    proba (float): The predicted probability.

    Returns:
    int: The assigned score.
    """
    score = (1 - proba[0]) * 100
    return int(score)


def main():
    # Charger le modèle XGBoost
    xgboost_model = load_model("xgboost_model.pkl")

    # Charger les données des individus à prédire
    individual_data = pd.read_csv("individual_test.csv", index_col=0)

    individual_data_processed = preprocess_data(individual_data)
    scores = []
    # Prédire pour chaque individu
    for index, row in individual_data_processed.iterrows():
        # Prédire la probabilité de défaut pour l'individu
        proba = predict_proba_individual(xgboost_model, row.to_numpy().reshape(1, -1))
        score = assign_score(proba)
        # Afficher la probabilité prédite
        print(f"Individu {index+1} - Probabilité de défaut : {proba} | -> {score}")

if __name__ == "__main__":
    main()