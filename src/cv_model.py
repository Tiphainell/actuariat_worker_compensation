import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold






def run_ablation_experiment(num_experiment: int,features_x_category: dict,feature_y: str,df : pd.DataFrame):
    """
        Réalise une ablation study sur différentes configurations de features.

        Cette fonction compare plusieurs ensembles de variables explicatives
        afin d'évaluer leur impact sur la performance du modèle.

        Chaque configuration est évaluée via une validation croisée KFold.

        Pour les modèles non-baseline, une pondération des observations
        est appliquée afin de donner plus d'importance aux sinistres extrêmes.

        Parameters
        ----------
        num_experiment : int
            Nombre de folds pour la validation croisée.

        features_x_category : dict
            Dictionnaire où :
            - clé = nom de la configuration (ex: "demographic", "nlp", etc.)
            - valeur = liste des variables utilisées pour le modèle

        feature_y : str
            Nom de la variable cible dans le DataFrame.

        df : pd.DataFrame
            Dataset complet contenant features et target.

        Returns
        -------
        metrics : dict
            Dictionnaire contenant RMSE et R² pour chaque configuration et fold.

        models : dict
            Modèles entraînés pour chaque fold et chaque configuration.
        """



    metrics = {}
    models = {}

    y = df[feature_y]

    # mêmes splits pour toutes les catégories
    kf = KFold(
        n_splits=num_experiment,
        shuffle=True,
        random_state=42
    )

    splits = list(kf.split(df))

    for category_name, input_variables in features_x_category.items():

        print(f"\n--- Testing category: {category_name} ---")

        X = df[input_variables]


        folds_rmse = {}
        folds_R2 = {}
        folds_model = {}

        for fold, (train_idx, val_idx) in enumerate(splits):

            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            if category_name != "baseline":

                weights = np.where(
                    y_tr > np.quantile(y_tr, 0.95),
                    2,
                    1
                )

                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42 + fold
                )

                model.fit(
                    X_tr,
                    np.log1p(y_tr),
                    sample_weight=weights
                )

                y_pred = np.expm1(model.predict(X_val))
            else:

                y_pred = X_val
                model = 1

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            folds_rmse[fold] = rmse
            folds_R2[fold] = r2
            folds_model[fold] = model

        metrics[category_name] = {
            "RMSE": folds_rmse,
            "R2": folds_R2
        }

        models[category_name] = folds_model

    return metrics, models

def train_final_model(df, input_features_x, feature_y):
    """
       Entraîne le modèle final XGBoost sur l'ensemble des données d'entraînement.

       Le modèle est entraîné sur une transformation log1p de la variable cible
       afin de limiter l'impact des valeurs extrêmes.

       Une pondération est appliquée pour accorder plus d'importance aux sinistres
       les plus élevés (top 5% de la distribution).

       Parameters
       ----------
       df : pd.DataFrame
           Dataset d'entraînement contenant features et target.

       input_features_x : list[str]
           Liste des variables explicatives utilisées pour l'entraînement.

       feature_y : str
           Nom de la variable cible.

       Returns
       -------
       model : xgboost.XGBRegressor
           Modèle XGBoost entraîné prêt pour la prédiction.
       """

    X_tr=df[input_features_x]
    y_tr = df[feature_y]
    weights = np.where(
        y_tr > np.quantile(y_tr, 0.95),
        2,
        1
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    model.fit(
        X_tr,
        np.log1p(y_tr),
        sample_weight=weights
    )

    return model


def run_xgb_gridsearch(X, y, use_log=True):
    """
        Effectue une recherche d'hyperparamètres (GridSearchCV) sur un modèle XGBoost.

        Cette fonction entraîne un modèle XGBoost en testant plusieurs combinaisons
        d'hyperparamètres via validation croisée KFold.

        Le modèle peut être entraîné sur la variable cible transformée par log1p
        afin de stabiliser la variance et réduire l'effet des outliers.

        Parameters
        ----------
        X : pd.DataFrame ou np.ndarray
            Variables explicatives du jeu de données.

        y : array-like
            Variable cible (ex: coût des sinistres).

        use_log : bool, default=True
            Si True, applique une transformation log1p sur la cible avant entraînement.

        Returns
        -------
        grid : GridSearchCV
            Objet GridSearchCV entraîné contenant les résultats de recherche.

        best_model : xgboost.XGBRegressor
            Meilleur modèle entraîné selon la validation croisée.
        """

    # -----------------------
    # 1. modèle de base
    # -----------------------
    model = xgb.XGBRegressor(
        random_state=42
    )

    # -----------------------
    # 2. grille hyperparamètres
    # -----------------------
    param_grid = {
        "n_estimators": [5, 10 ,100],
        "max_depth": [ 6, 10],
    }

    # -----------------------
    # 3. cross-validation
    # -----------------------
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # -----------------------
    # 4. GridSearchCV
    # -----------------------
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    # -----------------------
    # 5. fit (log transform optionnel)
    # -----------------------
    if use_log:
        y_train = np.log1p(y)
    else:
        y_train = y

    grid.fit(X, y_train)

    # -----------------------
    # 6. résultats
    # -----------------------
    print("Best params:", grid.best_params_)
    print("Best CV RMSE:", -grid.best_score_)

    best_model = grid.best_estimator_

    return grid, best_model