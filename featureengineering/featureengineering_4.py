import pandas as pd
import numpy as np
from pathlib import Path
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

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "ressources" / "reservations_train.csv"
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

# NaN dans agent_id = réservation directe
df["agent_id"] = df["agent_id"].fillna("direct")


# ============================================================
# FEATURE ENGINEERING — PRIX
# ============================================================

# Prix théorique du séjour à partir du prix moyen par nuit
df["prix_total_sejour"] = (
    df["prix_moyen_nuit_eur"] *
    df["nuits"]
)


# Différence entre le montant total et le prix
# calculé à partir du prix moyen par nuit.
df["ecart_prix_total"] = (
    df["montant_total_eur"] -
    df["prix_total_sejour"]
)


# Montant moyen réellement payé par nuit
df["montant_par_nuit"] = (
    df["montant_total_eur"] /
    df["nuits"].replace(0, np.nan)
)


# ============================================================
# CIBLE
# ============================================================

target = "reservation_annulee"


# ============================================================
# VARIABLES NUMERIQUES
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

    # Nouvelles features prix
    "prix_total_sejour",
    "ecart_prix_total",
    "montant_par_nuit"
]


# ============================================================
# VARIABLES CATEGORIELLES
# ============================================================

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

X = df[
    numeriques + categoriels
]

y = df[target]


# ============================================================
# SPLIT TEMPOREL
# ============================================================

# On garde exactement le même protocole que la baseline.
# La date de réservation détermine l'ordre temporel.

ordre = (
    df["date_reservation"]
    .sort_values()
    .index
)


X = X.loc[ordre]
y = y.loc[ordre]

df_ordre = df.loc[ordre]


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
    df_ordre["date_reservation"].iloc[0],
    "→",
    df_ordre["date_reservation"].iloc[n_train - 1]
)

print(
    "Période validation :",
    df_ordre["date_reservation"].iloc[n_train],
    "→",
    df_ordre["date_reservation"].iloc[-1]
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
# METRIQUES
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


# ============================================================
# RESULTATS
# ============================================================

print("\n" + "=" * 60)
print("RESULTATS — FEATURES PRIX")
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

f1_baseline = 0.4729

gain = f1 - f1_baseline


print("\n" + "=" * 60)
print("COMPARAISON AVEC BASELINE")
print("=" * 60)

print(
    f"F1 baseline : {f1_baseline:.4f}"
)

print(
    f"F1 prix     : {f1:.4f}"
)

print(
    f"Gain        : {gain:+.4f}"
)