import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

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
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(
    BASE_DIR / "ressources" / "reservations_train.csv"
)
# Dans l'énoncé :
# NaN dans agent_id = réservation directe
df["agent_id"] = df["agent_id"].fillna("DIRECT")


# ============================================================
# DATES
# ============================================================

df["date_reservation"] = pd.to_datetime(
    df["date_reservation"]
)

df["date_arrivee"] = pd.to_datetime(
    df["date_arrivee"]
)


# ============================================================
# TRI TEMPOREL
# ============================================================

df = df.sort_values(
    "date_reservation"
).reset_index(drop=True)


# ============================================================
# VARIABLES
# ============================================================

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
    "jours_liste_attente"
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
# X / y
# ============================================================

X = df[numeriques + categoriels]
y = df["reservation_annulee"]


# ============================================================
# SPLIT TEMPOREL
# ============================================================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_val = X.iloc[split:]

y_train = y.iloc[:split]
y_val = y.iloc[split:]


print("=" * 60)
print("SPLIT TEMPOREL")
print("=" * 60)

print("Nombre de données :", len(df))
print("Train :", len(X_train))
print("Validation :", len(X_val))

print(
    "\nPériode train :",
    df["date_reservation"].iloc[0],
    "→",
    df["date_reservation"].iloc[split - 1]
)

print(
    "Période validation :",
    df["date_reservation"].iloc[split],
    "→",
    df["date_reservation"].iloc[-1]
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeriques",
            Pipeline([
                (
                    "imputation",
                    SimpleImputer(strategy="median")
                ),
                (
                    "standardisation",
                    StandardScaler()
                )
            ]),
            numeriques
        ),

        (
            "categoriels",
            Pipeline([
                (
                    "imputation",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encodage",
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
# MODELE
# ============================================================

modele = Pipeline([
    ("preprocessing", preprocessor),

    (
        "classification",
        LogisticRegression(
            max_iter=2000
        )
    )
])


# ============================================================
# ENTRAINEMENT
# ============================================================

print("\n" + "=" * 60)
print("ENTRAINEMENT")
print("=" * 60)

modele.fit(X_train, y_train)

print("Entraînement terminé.")


# ============================================================
# PROBABILITES
# ============================================================

y_proba = modele.predict_proba(X_val)[:, 1]


# ============================================================
# DECISION — SEUIL 0.3
# ============================================================

seuil = 0.185

y_pred = (y_proba >= seuil).astype(int)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Seuil    : {seuil}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_val, y_pred))

print("\nClassification report :")
print(
    classification_report(
        y_val,
        y_pred,
        digits=4
    )
)

# ============================================================
# RECHERCHE DU MEILLEUR SEUIL
# ============================================================

seuils = np.arange(0.10, 0.30, 0.001)

resultats = []

for seuil in seuils:

    y_pred = (y_proba >= seuil).astype(int)

    resultats.append({
        "seuil": seuil,
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    })

resultats = pd.DataFrame(resultats)

meilleur = resultats.loc[
    resultats["f1"].idxmax()
]

print("\n" + "=" * 60)
print("MEILLEUR SEUIL")
print("=" * 60)

print(f"Seuil    : {meilleur['seuil']:.3f}")
print(f"Precision : {meilleur['precision']:.4f}")
print(f"Recall    : {meilleur['recall']:.4f}")
print(f"F1        : {meilleur['f1']:.4f}")