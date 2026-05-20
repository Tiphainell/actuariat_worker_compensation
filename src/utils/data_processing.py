import pandas as pd
import numpy as np
from src.utils.nlp_processing import add_nlp_columns
from typing import Tuple, List

def load_data_base(directory_path: str) -> pd.DataFrame:
    """
        Charge les jeux de données train et test puis les concatène
        dans une unique base de travail pour pouvoir ajouter les features sur tout le set.

        Un indicateur `role` est ajouté afin de distinguer les
        observations du train et du test.

        Parameters
        ----------
        directory_path : str
            Chemin vers le dossier contenant les fichiers CSV.

        Returns
        -------
        pd.DataFrame
            Base concaténée contenant train et test.
        """
    # 54 000 lignes pour train (60% échantillon)
    train = pd.read_csv(directory_path + '/train.csv', sep=",")
    # 36 000 lignes (40%) pour le test
    test = pd.read_csv(directory_path + '/test.csv', sep=",")
    # On doit fournir ClaimNumber et UltimateIncurredClaimCost
    # toute la base est rassembléee pour faire les traitements qu'une fois
    database = pd.concat([train.assign(role="train"), test.assign(role="test")])

    return database

def add_features(database: pd.DataFrame,nlp: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
        Applique l'ensemble des étapes de feature engineering.

        Les transformations incluent :
        - le formatage des variables catégorielles,
        - l'ajout de variables démographiques et temporelles,
        - l'ajout optionnel de variables issues du NLP.

        Deux listes de features sont également construites :
        - une liste contenant uniquement les features démographiques,
        - une liste contenant les features démographiques + NLP.

        Parameters
        ----------
        database : pd.DataFrame
            Base de données complète contenant train et test.
        nlp : bool
            Indique si les features NLP doivent être générées.

        Returns
        -------
        tuple[pd.DataFrame, list[str], list[str]]
            - Base enrichie avec les nouvelles features
            - Liste des features démographiques
            - Liste des features démographiques + NLP
        """

    database = format_features(database)
    # add demographic_features
    database = add_demog_features(database)

    cols_to_remove = [
        "ClaimNumber",
        "DateTimeOfAccident",
        "UltimateIncurredClaimCost",
        "DateReported",
        "ClaimDescription",
        "ClaimDescriptionClean",
        "role",
        "claim_development_ratio"
    ]
    list_features_demog = database.columns.to_list()

    list_features_demog = [
        col for col in list_features_demog
        if col not in cols_to_remove
    ]

    list_features_demog_nlp=[]

    if nlp:

        database = add_nlp_columns(database)

        list_features_demog_nlp = database.columns.to_list()
        list_features_demog_nlp= [
            col for col in list_features_demog_nlp
            if col not in cols_to_remove
        ]




    return database, list_features_demog, list_features_demog_nlp


def format_features(database: pd.DataFrame) -> pd.DataFrame:
    """
        Encode les variables catégorielles sous forme numérique.

        Les variables catégorielles sont transformées via un
        one-hot encoding afin d'être exploitables par les modèles
        de machine learning.

        Certaines modalités peuvent correspondre à des valeurs
        inconnues ou non renseignées (`U` : Undefined).

        Parameters
        ----------
        database : pd.DataFrame
            Base de données à transformer.

        Returns
        -------
        pd.DataFrame
            Base contenant les variables catégorielles encodées.
        """
    # On encode en one-hot encoding les variables catégorielles (certaines catégories sont à U : Undefined)
    database = pd.get_dummies(database, columns=["Gender"], drop_first=True)
    database = pd.get_dummies(database, columns=["MaritalStatus"], drop_first=True)
    database = pd.get_dummies(database, columns=["PartTimeFullTime"], drop_first=True)



    return database

def add_demog_features(database: pd.DataFrame) -> pd.DataFrame:
    """
       Génère des variables démographiques, temporelles et métier.

       Les features ajoutées incluent notamment :
       - des composantes temporelles liées à la date de l'accident,
       - le délai de déclaration du sinistre,
       - des indicateurs liés aux salaires,
       - des ratios métier,
       - des transformations non linéaires sur l'âge.

       Certaines variables sont construites afin d'aider le modèle
       à capturer des effets d'inflation ou de temporalité.

       Parameters
       ----------
       database : pd.DataFrame
           Base de données contenant les variables brutes.

       Returns
       -------
       pd.DataFrame
           Base enrichie avec les nouvelles features démographiques
           et temporelles.
       """

    # récupération de l'année, mois, jour, semaine, heure de l'accident pour que le modèle puisse exploiter la temporalité pour l'inflation
    database["DateTimeOfAccident"] = pd.to_datetime(database["DateTimeOfAccident"])
    database["DateTimeOfAccident_year"] = database["DateTimeOfAccident"].dt.year
    database["DateTimeOfAccident_month"] = database["DateTimeOfAccident"].dt.month
    database["DateTimeOfAccident_day"] = database["DateTimeOfAccident"].dt.day
    database["DateTimeOfAccident_hour"] = database["DateTimeOfAccident"].dt.hour
    # la colonne minute n'apporte rien, je l'enlève (tjs à 0)
    # database["DateTimeOfAccident_minute"] = database["DateTimeOfAccident"].dt.minute
    database["DateTimeOfAccident_week"] = database["DateTimeOfAccident"].dt.isocalendar().week

    # On calcule le delta entre le jour de l'accident et la date reported
    database["reporting_delay"] = (pd.to_datetime(database["DateReported"]) - database["DateTimeOfAccident"]).dt.days

    #ajout de variables pour essayer de capturer l'inflation des salaires
    database["wages_per_hour"] = np.where(
        database["HoursWorkedPerWeek"] > 0,
        database["WeeklyWages"] / database["HoursWorkedPerWeek"],
        database["WeeklyWages"]/(database["DaysWorkedPerWeek"]*8)
    )

    database["claim_given_wages"] = database["InitialIncurredCalimsCost"] / database["wages_per_hour"]


    #calcul du wage relatif par rapport à l'année :
    median_wage_by_year = (
        database[database['role']=="train"].groupby("DateTimeOfAccident_year")["WeeklyWages"].median()
    )
    database["relative_wage"] = (
            database["wages_per_hour"] /
            database["DateTimeOfAccident_year"].map(median_wage_by_year)
    )


    #dependance en age 2

    database["age_2"] = database["Age"]*database["Age"]


    return database

