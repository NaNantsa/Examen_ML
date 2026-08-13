import pandas as pd
import joblib


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

modele = joblib.load("modele_random_forest.pkl")
seuil = joblib.load("seuil_random_forest.pkl")

print("=" * 60)
print("MODELE")
print("=" * 60)

print("Modèle chargé.")
print(f"Seuil optimal : {seuil:.3f}")


# ============================================================
# CHARGEMENT DU TEST
# ============================================================

df_test = pd.read_csv(
    "../ressources/reservations_test.csv"
)

print("\nNombre de réservations test :", len(df_test))


# ============================================================
# PREPARATION
# ============================================================

df_test["date_reservation"] = pd.to_datetime(
    df_test["date_reservation"]
)

df_test["date_arrivee"] = pd.to_datetime(
    df_test["date_arrivee"]
)

# NaN dans agent_id = réservation directe
df_test["agent_id"] = df_test["agent_id"].fillna("direct")

# Variables temporelles
df_test["mois_arrivee"] = df_test["date_arrivee"].dt.month
df_test["mois_reservation"] = df_test["date_reservation"].dt.month

# Suppression des dates brutes
df_test = df_test.drop(
    columns=[
        "date_reservation",
        "date_arrivee"
    ]
)

# On conserve l'identifiant pour le fichier final
reservation_ids = df_test["reservation_id"]

df_test = df_test.drop(
    columns=["reservation_id"]
)


# ============================================================
# PREDICTION
# ============================================================

probabilites = modele.predict_proba(
    df_test
)[:, 1]

predictions = (
    probabilites >= seuil
).astype(int)


# ============================================================
# CREATION DU FICHIER FINAL
# ============================================================

submission = pd.DataFrame({
    "reservation_id": reservation_ids,
    "reservation_annulee": predictions,
    "probabilite_annulation": probabilites
})


# ============================================================
# SAUVEGARDE
# ============================================================

submission.to_csv(
    "submission.csv",
    index=False
)


# ============================================================
# RESULTATS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION TERMINEE")
print("=" * 60)

print("Fichier créé : submission.csv")

print("\nAperçu :")
print(submission.head(10))

print("\nNombre de lignes :", len(submission))