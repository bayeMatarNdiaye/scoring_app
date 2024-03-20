import streamlit as st
import pandas as pd
from predict import load_model, preprocess_data, predict_proba_individual, assign_score 



def main():
    st.title("Outil d'attribution de Score")

    st.write(

        """
        Bienvenue dans notre application d'attribution de Score Bancaire. Cet outil est conçu pour vous aider dans votre processus de prise de décision concernant l'octroi de crédit. Veuillez noter que le score attribué par notre modèle ne garantit ni l'acceptation ni le refus d'une demande de crédit, mais fournit une indication pour éclairer votre décision finale.

        Pour utiliser notre application, veuillez télécharger un fichier contenant les informations des individus de votre portefeuille clients. Notre modèle analysera ces données et attribuera un score à chaque individu, fournissant ainsi un aperçu de leur crédibilité financière.
        """
    )
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, index_col=0)
        st.write("Données :")
        st.write(data)
        
        
        xgboost_model = load_model("xgboost_model.pkl")


        data_processed = preprocess_data(data)


        predictions = []
        for index, row in data_processed.iterrows():
            
            proba = predict_proba_individual(xgboost_model, row.to_numpy().reshape(1, -1))
            score = assign_score(proba)
            predictions.append({"Probabilité de défaut": proba[0], "Score": score})

        #scores
        st.write("Résultats :")
        predictions_df = pd.DataFrame(predictions)
        st.write(predictions_df)

if __name__ == "__main__":
    main()
