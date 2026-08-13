import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    "ressources/reservations_train.csv"
)

df["date_reservation"] = pd.to_datetime(
    df["date_reservation"]
)

df["date_arrivee"] = pd.to_datetime(
    df["date_arrivee"]
)


# ============================================================
# PREPARATION
# ============================================================

# IMPORTANT :
# NaN dans agent_id = réservation directe
df["agent_id"] = df["agent_id"].fillna("direct")


# ============================================================
# ORDRE TEMPOREL
# ============================================================

# Même méthode que dans le baseline :
# on trie les indices selon la date d'arrivée.

ordre = (
    df["date_arrivee"]
    .sort_values()
    .index
)


# On applique cet ordre à toutes les lignes
df = df.loc[ordre].reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING : DATES
# ============================================================

df["mois_arrivee"] = (
    df["date_arrivee"].dt.month
)

df["mois_reservation"] = (
    df["date_reservation"].dt.month
)

df["jour_semaine_arrivee"] = (
    df["date_arrivee"].dt.dayofweek
)

# 0 = lundi
# 5 = samedi
# 6 = dimanche

df["weekend_arrivee"] = (
    df["jour_semaine_arrivee"] >= 5
).astype(int)


def determiner_saison(mois):

    if mois in [12, 1, 2]:
        return "hiver"

    elif mois in [3, 4, 5]:
        return "printemps"

    elif mois in [6, 7, 8]:
        return "ete"

    else:
        return "automne"


df["saison_arrivee"] = (
    df["mois_arrivee"]
    .apply(determiner_saison)
)


# ============================================================
# CIBLE
# ============================================================

target = "reservation_annulee"

X = df.drop(
    columns=[
        target,
        "reservation_id",
        "date_reservation",
        "date_arrivee"
    ]
)

y = df[target]


# ============================================================
# VARIABLES
# ============================================================

numeriques = [

    # Variables originales
    "categorie_hotel",
    "delai_reservation_jours",
    "nuits",
    "adultes",
    "enfants",
    "chambres",
    "prix_moyen_nuit_eur",
    "remise_pct",
    "montant_total_eur",
    "reservations_passees",
    "annulations_passees",
    "demandes_speciales",
    "modifications_reservation",
    "jours_liste_attente",

    # Features dates
    "mois_arrivee",
    "mois_reservation",
    "jour_semaine_arrivee",
    "weekend_arrivee"
]


categoriels = [

    # Variables originales
    "region_hotel",
    "ville",
    "type_destination",
    "hotel_id",
    "segment_client",
    "marche_origine",
    "canal_reservation",
    "moyen_transport",
    "formule_repas",
    "tarif_remboursable",
    "type_acompte",
    "client_type",
    "agent_id",

    # Feature date
    "saison_arrivee"
]


# ============================================================
# SPLIT TEMPOREL
# ============================================================

n_train = int(
    len(X) * 0.80
)

X_train = X.iloc[:n_train]
X_val = X.iloc[n_train:]

y_train = y.iloc[:n_train]
y_val = y.iloc[n_train:]


print("=" * 60)
print("SPLIT TEMPOREL")
print("=" * 60)

print(
    "Nombre de données :",
    len(X)
)

print(
    "Train :",
    len(X_train)
)

print(
    "Validation :",
    len(X_val)
)

print(
    "\nPériode train :",
    df["date_arrivee"].iloc[0],
    "→",
    df["date_arrivee"].iloc[n_train - 1]
)

print(
    "Période validation :",
    df["date_arrivee"].iloc[n_train],
    "→",
    df["date_arrivee"].iloc[-1]
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocesseur = ColumnTransformer(

    transformers=[

        (
            "numeriques",

            Pipeline([
                (
                    "imputation",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),

            numeriques
        ),

        (
            "categoriels",

            Pipeline([

                (
                    "imputation",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),

            categoriels
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

modele = Pipeline([

    (
        "preprocessing",
        preprocesseur
    ),

    (
        "classifier",

        RandomForestClassifier(

            n_estimators=300,

            random_state=42,

            n_jobs=-1,

            class_weight=None
        )
    )
])


# ============================================================
# ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT")
print("=" * 60)

modele.fit(
    X_train,
    y_train
)

print(
    "Entraînement terminé."
)


# ============================================================
# PROBABILITES
# ============================================================

y_proba = modele.predict_proba(
    X_val
)[:, 1]


# ============================================================
# RECHERCHE DU MEILLEUR SEUIL
# ============================================================

seuils = np.arange(
    0.05,
    0.96,
    0.001
)

resultats = []


for seuil in seuils:

    y_pred = (
        y_proba >= seuil
    ).astype(int)

    resultats.append({

        "seuil": seuil,

        "precision": precision_score(
            y_val,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_val,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_val,
            y_pred,
            zero_division=0
        )
    })


resultats = pd.DataFrame(
    resultats
)


# ============================================================
# MEILLEUR SEUIL
# ============================================================

meilleur = resultats.loc[
    resultats["f1"].idxmax()
]

seuil_optimal = meilleur["seuil"]


y_pred = (
    y_proba >= seuil_optimal
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

precision = precision_score(
    y_val,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("RESULTATS — FEATURES DATES")
print("=" * 60)

print(
    f"Seuil    : {seuil_optimal:.3f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)


# ============================================================
# MATRICE DE CONFUSION
# ============================================================

print("\nMatrice de confusion :")

print(
    confusion_matrix(
        y_val,
        y_pred
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification report :")

print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# COMPARAISON AVEC BASELINE
# ============================================================

f1_baseline = 0.4900

gain = f1 - f1_baseline


print("\n" + "=" * 60)
print("COMPARAISON AVEC BASELINE")
print("=" * 60)

print(
    f"F1 baseline : {f1_baseline:.4f}"
)

print(
    f"F1 dates    : {f1:.4f}"
)

print(
    f"Gain        : {gain:+.4f}"
)