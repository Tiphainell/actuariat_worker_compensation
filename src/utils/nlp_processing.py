import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
nltk.download('stopwords')
stops = stopwords.words('english')

PORTER_STEMMER = PorterStemmer()



# Dictionnaire de mots-clés associés aux différentes parties du corps
BODY_PART_KEYWORDS = {
    "head": {"head", "skull", "brain"},
    "eye": {"eye", "vision"},
    "neck": {"neck", "cervic"},
    "shoulder": {"shoulder"},
    "arm": {"arm", "elbow", "forearm"},
    "hand": {"wrist", "hand", "finger", "thumb", "index"},
    "back": {"back", "spine", "lumbar", "cervic"},
    "chest": {"chest", "thorac"},
    "abdomen": {"abdomen", "stomach", "abdomin"},
    "hip": {"hip", "pelvis"},
    "knee": {"knee"},
    "foot": {"ankl", "foot", "toe"},
    "tissu": {"tissu"}
}

# Dictionnaire de mots-clés associés aux différents types de blessures
INJURY_TYPE_KEYWORDS = {
    "fracture": {"fractur", "fracture"},
    "strain": {"strain"},
    "sprain": {"sprain"},
    "laceration": {"lacer"},
    "contusion": {"contusion", "bruise"}
}



def text_clean(claim:str)->list[str]:
    """
        Nettoie et normalise une description de sinistre.

        Les étapes de preprocessing appliquées sont :
        - conversion en minuscules,
        - tokenisation simple par séparation des espaces,
        - suppression des stopwords anglais,
        - stemming des mots via PorterStemmer.

        Cette transformation permet de réduire la variabilité
        textuelle avant l'extraction de features NLP.

        Parameters
        ----------
        claim : str
            Description textuelle du sinistre.

        Returns
        -------
        str
            Liste des tokens nettoyés et stemmés.
        """


    # Converting to Lower Case
    claim =claim.lower()

    # Getting List Of Words
    claim =claim.split()

    # Removing Stop Words(Words which do not add any information like =is,are,I etc)
    claim =[word for word in claim if word not in stops]

    # Stemming the word(words like playing ,played are replaced with play)
    claim = [PORTER_STEMMER.stem(word) for word in claim]


    return claim


def get_injury_type(text):
    """
        Identifie le type de blessure à partir de la description du sinistre.

        La détection repose sur des règles simples basées sur
        la présence de mots-clés ou de racines lexicales.

        Les catégories détectées incluent notamment :
        - fracture,
        - strain,
        - sprain,
        - laceration,
        - contusion.

        Parameters
        ----------
        text : str
            Texte nettoyé de la description du sinistre.

        Returns
        -------
        str
            Type de blessure détecté.
        """
    text = text.lower()

    text = text.lower()

    for injury_type, keywords in INJURY_TYPE_KEYWORDS.items():

        if any(keyword in text for keyword in keywords):
            return injury_type

    return "other"


def get_body_part(text):
    """
        Identifie la partie du corps concernée par le sinistre.

        La détection est réalisée à partir de règles simples
        basées sur des mots-clés présents dans le texte.

        Les catégories détectées incluent notamment :
        - head,
        - neck,
        - back,
        - hand,
        - knee,
        - foot.

        Parameters
        ----------
        text : str
            Texte nettoyé de la description du sinistre.

        Returns
        -------
        str
            Partie du corps détectée.
        """
    text = text.lower()

    text = text.lower()

    for body_part, keywords in BODY_PART_KEYWORDS.items():

        if any(keyword in text for keyword in keywords):
            return body_part

    return "other"

def get_laterality(text):
    """
        Détecte la latéralité mentionnée dans la description du sinistre.

        Les catégories possibles sont :
        - left,
        - right,
        - bilateral,
        - unknown.

        Parameters
        ----------
        text : str
            Texte nettoyé de la description du sinistre.

        Returns
        -------
        str
            Latéralité détectée.
        """
    text = text.lower()

    has_left = "left" in text or "lft" in text
    has_right = "right" in text or "rght" in text

    if has_left and has_right:
        return "bilateral"
    elif has_left:
        return "left"
    elif has_right:
        return "right"
    else:
        return "unknown"


def get_stress(text):
    """
        Détecte la présence du terme 'stress' dans la description.

        Cette variable est utilisée comme indicateur binaire.

        Parameters
        ----------
        text : str
            Texte nettoyé de la description du sinistre.

        Returns
        -------
        int
            1 si le mot 'stress' est détecté, sinon 0.
        """
    text = text.lower()

    if "stress" in text:
        return 1
    else:
        return 0

def add_nlp_columns(database):

    """
    Génère des features NLP à partir des descriptions de sinistres.

    Les transformations réalisées incluent :
    - le nettoyage des descriptions textuelles,
    - l'extraction de variables métier simples :
        - type de blessure,
        - partie du corps touchée,
        - latéralité,
        - présence de stress,
    - l'encodage one-hot des variables catégorielles extraites.

    Cette approche repose sur des règles heuristiques simples
    afin d'enrichir les données tabulaires avec de l'information
    issue du texte.

    Parameters
    ----------
    database : pd.DataFrame
        Base de données contenant la colonne `ClaimDescription`.

    Returns
    -------
    pd.DataFrame
        Base enrichie avec les nouvelles features NLP.
    """

    database['ClaimDescriptionClean'] = database['ClaimDescription'].apply(lambda x: ' '.join(text_clean(x)))
    database["laterality"] = database["ClaimDescriptionClean"].apply(get_laterality)
    database["body_part"] = database["ClaimDescriptionClean"].apply(get_body_part)
    database["injury_type"] = database["ClaimDescriptionClean"].apply(get_injury_type)
    database["stress"] = database["ClaimDescriptionClean"].apply(get_stress)
    #database["combined_laterality_and_body_part"] = (
    #        database["laterality"].astype(str) + "_" + database["body_part"].astype(str)
    #)
    #database["year_injury_type"]=(database["injury_type"].astype(str) + "_" + database["DateTimeOfAccident_year"].astype(str))

    database = pd.get_dummies(
        database,
        columns=["laterality", "body_part", "injury_type"],
        drop_first=True
    )

    return database