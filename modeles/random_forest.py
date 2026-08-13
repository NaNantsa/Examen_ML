import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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

df = pd.read_csv("ressources/reservations_train.csv")

df["date_reservation"] = pd.to_datetime(df["date_reservation"])
df["date_arrivee"] = pd.to_datetime(df["date_arrivee"])


# ============================================================
# PREPARATION
# ============================================================

# IMPORTANT :
# NaN dans agent_id = réservation directe
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
df = df.drop(columns=["reservation_id"])


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

ordre = pd.to_datetime(
    pd.read_csv("ressources/reservations_train.csv")["date_reservation"]
).sort_values().index

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


# ============================================================
# PREPROCESSING
# ============================================================

preprocesseur = ColumnTransformer(
    transformers=[
        (
            "numeriques",
            Pipeline([
                ("imputation", SimpleImputer(strategy="median"))
            ]),
            numeriques
        ),
        (
            "categoriels",
            Pipeline([
                ("imputation", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore"
                ))
            ]),
            categoriels
        )
    ]
)


# ============================================================
# MODELE
# ============================================================

modele = Pipeline([
    ("preprocessing", preprocesseur),
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

modele.fit(X_train, y_train)

print("Entraînement terminé.")


# ============================================================
# PROBABILITES
# ============================================================

y_proba = modele.predict_proba(X_val)[:, 1]


# ============================================================
# RECHERCHE DU MEILLEUR SEUIL
# ============================================================

seuils = np.arange(0.05, 0.96, 0.001)

resultats = []

for seuil in seuils:

    y_pred = (y_proba >= seuil).astype(int)

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


resultats = pd.DataFrame(resultats)

meilleur = resultats.loc[
    resultats["f1"].idxmax()
]


# ============================================================
# EVALUATION
# ============================================================

seuil_optimal = meilleur["seuil"]

y_pred = (y_proba >= seuil_optimal).astype(int)

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

print(f"Seuil    : {seuil_optimal:.3f}")
print(f"Precision : {precision_score(y_val, y_pred):.4f}")
print(f"Recall    : {recall_score(y_val, y_pred):.4f}")
print(f"F1        : {f1_score(y_val, y_pred):.4f}")

print("\nMatrice de confusion :")
print(confusion_matrix(y_val, y_pred))

print("\nClassification report :")
print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# IMPORTANCE DES VARIABLES
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANCE DES VARIABLES")
print("=" * 60)


# Récupération du Random Forest
random_forest = modele.named_steps["classifier"]


# Récupération des noms après One-Hot Encoding
noms_variables = (
    modele
    .named_steps["preprocessing"]
    .get_feature_names_out()
)


# Importance de chaque variable transformée
importances = random_forest.feature_importances_


importance_detail = pd.DataFrame({
    "variable": noms_variables,
    "importance": importances
})


importance_detail = importance_detail.sort_values(
    "importance",
    ascending=False
)


print("\nTop 20 variables transformées :")
print(
    importance_detail.head(20).to_string(index=False)
)


# ============================================================
# REGROUPEMENT PAR VARIABLE ORIGINALE
# ============================================================

importance_originale = {}

for _, ligne in importance_detail.iterrows():

    nom = ligne["variable"]
    importance = ligne["importance"]

    # Exemple :
    # numeriques__delai_reservation_jours
    # categoriels__type_acompte_total

    if "__" in nom:
        nom_sans_prefixe = nom.split("__", 1)[1]
    else:
        nom_sans_prefixe = nom

    # Pour les variables catégorielles One-Hot,
    # on doit retrouver la variable originale.
    variable_originale = None

    for col in numeriques + categoriels:

        if nom_sans_prefixe == col:
            variable_originale = col
            break

        if nom_sans_prefixe.startswith(col + "_"):
            variable_originale = col
            break

    if variable_originale is None:
        variable_originale = nom_sans_prefixe

    importance_originale[variable_originale] = (
        importance_originale.get(variable_originale, 0)
        + importance
    )


importance_originale = pd.DataFrame(
    importance_originale.items(),
    columns=["variable", "importance"]
)


importance_originale = importance_originale.sort_values(
    "importance",
    ascending=False
)


# ============================================================
# RESULTATS
# ============================================================

print("\n" + "=" * 60)
print("IMPORTANCE DES VARIABLES ORIGINALES")
print("=" * 60)

print(
    importance_originale.to_string(index=False)
)


# ============================================================
# TOP 15
# ============================================================

top = importance_originale.head(15)

print("\n" + "=" * 60)
print("TOP 15 VARIABLES")
print("=" * 60)

print(
    top.to_string(index=False)
)

# ============================================================
# SAUVEGARDE DU MODELE
# ============================================================

joblib.dump(
    modele,
    "modele_random_forest.pkl"
)

# Sauvegarde également du seuil optimal
joblib.dump(
    seuil_optimal,
    "seuil_random_forest.pkl"
)

print("\n" + "=" * 60)
print("SAUVEGARDE")
print("=" * 60)

print("Modèle sauvegardé : modele_random_forest.pkl")
print("Seuil sauvegardé  : seuil_random_forest.pkl")


# ============================================================
# GRAPHIQUE
# ============================================================

plt.figure(figsize=(10, 7))

plt.barh(
    top["variable"][::-1],
    top["importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Variable")
plt.title("Top 15 des variables les plus importantes")

plt.tight_layout()
plt.show()


# ============================================================
# ANALYSE DES ERREURS
# ============================================================

print("\n" + "=" * 60)
print("ANALYSE DES ERREURS")
print("=" * 60)


# ------------------------------------------------------------
# Faux positifs
# ------------------------------------------------------------

faux_positifs = X_val[
    (y_val == 0) & (y_pred == 1)
].copy()

print("\nNombre de faux positifs :", len(faux_positifs))


# ------------------------------------------------------------
# Faux négatifs
# ------------------------------------------------------------

faux_negatifs = X_val[
    (y_val == 1) & (y_pred == 0)
].copy()

print("Nombre de faux négatifs :", len(faux_negatifs))


# ============================================================
# AJOUT DES INFORMATIONS UTILES
# ============================================================

faux_positifs["probabilite_annulation"] = y_proba[
    (y_val == 0) & (y_pred == 1)
]

faux_negatifs["probabilite_annulation"] = y_proba[
    (y_val == 1) & (y_pred == 0)
]


# ============================================================
# AFFICHAGE
# ============================================================

colonnes_interessantes = [
    "hotel_id",
    "agent_id",
    "delai_reservation_jours",
    "prix_moyen_nuit_eur",
    "montant_total_eur",
    "marche_origine",
    "canal_reservation",
    "moyen_transport",
    "type_acompte",
    "tarif_remboursable",
    "segment_client",
    "probabilite_annulation"
]


print("\n" + "=" * 60)
print("EXEMPLES DE FAUX POSITIFS")
print("=" * 60)

print(
    faux_positifs[
        colonnes_interessantes
    ]
    .sort_values(
        "probabilite_annulation",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("EXEMPLES DE FAUX NEGATIFS")
print("=" * 60)

print(
    faux_negatifs[
        colonnes_interessantes
    ]
    .sort_values(
        "probabilite_annulation",
        ascending=True
    )
    .head(10)
    .to_string(index=False)
)