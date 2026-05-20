from src.utils.data_processing import load_data_base, add_features
from cv_model import train_final_model
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def main(data_path: str, nlp: bool):
    """
    Pipeline complet d'entraînement et d'inférence du modèle.

    Cette fonction réalise les étapes suivantes :
    - Chargement des données brutes
    - Feature engineering (avec option NLP)
    - Entraînement du modèle sur le jeu d'entraînement
    - Prédiction sur le jeu de test
    - Génération d'un fichier de soumission Kaggle

    Paramètres
    ----------
    data_path : str
        Chemin vers le dossier contenant les données.
    nlp : bool
        Indique si les features NLP doivent être ajoutées.

    Retour
    ------
    pd.DataFrame
        DataFrame contenant les prédictions finales pour la soumission.
    """

    # loading de toute la base
    database = load_data_base(data_path)

    # features engineering, ajout des features nlp si nlp=True
    database, list_features_x,_ = add_features(database, nlp)

    # Entrainement du modèle
    train = database[database["role"] == "train"]
    model = train_final_model(train, list_features_x, "UltimateIncurredClaimCost")

    #Prédiction sur le test set
    test = database[database['role'] == "test"]
    X_test = test[list_features_x]
    y_pred = np.expm1(model.predict(X_test))

    #Export des résultats pour kaggle
    results = pd.DataFrame({
        "ClaimNumber": test["ClaimNumber"],
        "UltimateIncurredClaimCost": y_pred})

    output_path = Path(__file__).resolve().parent.parent / "resultats" / "final_model.csv"
    results.to_csv(output_path, index=False)


if __name__ == '__main__':

    data_path = "/home/tiphainell/Documents/5.Direct Assurance/actuarial-loss-estimation"
    nlp = False
    main(data_path, nlp)
