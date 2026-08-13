import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
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

df = pd.read_csv("ressources/reservations_train.csv")

df["date_reservation"] = pd.to_datetime(
    df["date_reservation"]
)

df["date_arrivee"] = pd.to_datetime(
    df["date_arrivee"]
)


# ============================================================
# PREPARATION
# ============================================================

# agent_id = NaN signifie réservation directe
df["agent_id"] = df["agent_id"].fillna("direct")


# Variables temporelles
df["mois_arrivee"] = df["date_arrivee"].dt.month
df["mois_reservation"] = df["date_reservation"].dt.month


# Les dates brutes ne sont pas utilisées directement
df = df.drop(columns=[
    "date_reservation",
    "date_arrivee"
])


# reservation_id n'est pas une variable explicative
df = df.drop(columns=[
    "reservation_id"
])


# ============================================================
# VARIABLES
# ============================================================

target = "reservation_annulee"

X = df.drop(columns=[target])
y = df[target]


numeriques = [
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
    "mois_arrivee",
    "mois_reservation"
]


categoriels = [
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
    "agent_id"
]


# ============================================================
# SPLIT TEMPOREL
# ============================================================

# On trie selon la date d'arrivée.
# Le modèle apprend uniquement sur le passé
# et est évalué sur une période plus récente.

ordre = (
    pd.to_datetime(
        pd.read_csv(
            "ressources/reservations_train.csv"
        )["date_reservation"]
    )
    .sort_values()
    .index
)

X = X.loc[ordre]
y = y.loc[ordre]

n_train = int(len(X) * 0.80)

X_train = X.iloc[:n_train]
X_val = X.iloc[n_train:]

y_train = y.iloc[:n_train]
y_val = y.iloc[n_train:]


print("=" * 60)
print("SPLIT TEMPOREL")
print("=" * 60)

print("Nombre de données :", len(X))
print("Train :", len(X_train))
print("Validation :", len(X_val))

print(
    "\nPériode train :",
    X_train.index.min(),
    "→",
    X_train.index.max()
)

print(
    "Période validation :",
    X_val.index.min(),
    "→",
    X_val.index.max()
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocesseur = ColumnTransformer(
    transformers=[

        # -------------------------
        # Variables numériques
        # -------------------------
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

        # -------------------------
        # Variables catégorielles
        # -------------------------
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
                        handle_unknown="ignore",

                        # IMPORTANT :
                        # HistGradientBoosting nécessite
                        # une matrice dense.
                        sparse_output=False
                    )
                )
            ]),

            categoriels
        )
    ]
)


# ============================================================
# MODELE
# ============================================================

modele = Pipeline([

    (
        "preprocessing",
        preprocesseur
    ),

    (
        "classifier",

        HistGradientBoostingClassifier(

            max_iter=300,

            learning_rate=0.05,

            max_leaf_nodes=31,

            random_state=42
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

print("Entraînement terminé.")


# ============================================================
# PROBABILITES
# ============================================================

# Probabilité que reservation_annulee = 1

y_proba = modele.predict_proba(
    X_val
)[:, 1]


# ============================================================
# EVALUATION AU SEUIL 0.5
# ============================================================

seuil = 0.5

y_pred = (
    y_proba >= seuil
).astype(int)


print("\n" + "=" * 60)
print("EVALUATION — SEUIL 0.5")
print("=" * 60)

print(
    f"Precision : "
    f"{precision_score(y_val, y_pred, zero_division=0):.4f}"
)

print(
    f"Recall    : "
    f"{recall_score(y_val, y_pred, zero_division=0):.4f}"
)

print(
    f"F1        : "
    f"{f1_score(y_val, y_pred, zero_division=0):.4f}"
)


# ============================================================
# RECHERCHE DU MEILLEUR SEUIL
# ============================================================

print("\n" + "=" * 60)
print("RECHERCHE DU MEILLEUR SEUIL")
print("=" * 60)


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

    resultats.append({
        "seuil": seuil,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


resultats = pd.DataFrame(
    resultats
)


# Meilleur seuil selon le F1
meilleur = resultats.loc[
    resultats["f1"].idxmax()
]


seuil_optimal = meilleur["seuil"]


# ============================================================
# EVALUATION — SEUIL OPTIMAL
# ============================================================

y_pred = (
    y_proba >= seuil_optimal
).astype(int)


print("\n" + "=" * 60)
print("MEILLEUR SEUIL")
print("=" * 60)

print(
    f"Seuil    : "
    f"{seuil_optimal:.3f}"
)

print(
    f"Precision : "
    f"{precision_score(y_val, y_pred, zero_division=0):.4f}"
)

print(
    f"Recall    : "
    f"{recall_score(y_val, y_pred, zero_division=0):.4f}"
)

print(
    f"F1        : "
    f"{f1_score(y_val, y_pred, zero_division=0):.4f}"
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