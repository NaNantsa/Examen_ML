import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv("ressources/reservations_train.csv")
pd.set_option("display.max_columns", None)

print("=" * 60)
print("STRUCTURE")
print("=" * 60)
print("Dimensions :", df.shape)
print("Doublons lignes :", df.duplicated().sum())
print("Doublons reservation_id :", df["reservation_id"].duplicated().sum())


# ============================================================
# TRAITEMENT DE agent_id
# ============================================================
# Dans les données, NaN signifie que la réservation a été
# effectuée directement, sans passer par un agent.

df["agent_id"] = df["agent_id"].fillna("DIRECT")


# ============================================================
# CIBLE
# ============================================================

print("\n" + "=" * 60)
print("CIBLE : reservation_annulee")
print("=" * 60)

print(df["reservation_annulee"].value_counts())
print("\nProportions :")
print(df["reservation_annulee"].value_counts(normalize=True))

sns.countplot(data=df, x="reservation_annulee")
plt.title("Distribution de la cible")
plt.xlabel("Réservation annulée")
plt.ylabel("Nombre")


# ============================================================
# VALEURS MANQUANTES
# ============================================================

print("\n" + "=" * 60)
print("VALEURS MANQUANTES")
print("=" * 60)

missing = pd.DataFrame({
    "nombre": df.isna().sum(),
    "pourcentage": df.isna().mean() * 100
})

print(missing[missing["nombre"] > 0].sort_values("nombre", ascending=False))

missing_plot = missing[missing["nombre"] > 0].sort_values("pourcentage")

if not missing_plot.empty:
    missing_plot["pourcentage"].plot(
        kind="barh",
        figsize=(8, 4)
    )
    plt.title("Valeurs manquantes")
    plt.xlabel("Pourcentage")


# ============================================================
# DATES
# ============================================================

df["date_reservation"] = pd.to_datetime(
    df["date_reservation"],
    errors="coerce"
)

df["date_arrivee"] = pd.to_datetime(
    df["date_arrivee"],
    errors="coerce"
)

print("\nDates invalides :")
print("date_reservation :", df["date_reservation"].isna().sum())
print("date_arrivee :", df["date_arrivee"].isna().sum())

# Variables temporelles uniquement pour l'EDA
df["mois_arrivee"] = df["date_arrivee"].dt.month
df["mois_reservation"] = df["date_reservation"].dt.month


# ============================================================
# VARIABLES NUMERIQUES
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

print("\n" + "=" * 60)
print("STATISTIQUES NUMERIQUES")
print("=" * 60)

print(df[numeriques].describe().T)


# Distributions
df[numeriques].hist(
    figsize=(14, 10),
    bins=30
)

plt.tight_layout()


# ============================================================
# VALEURS ATYPIQUES
# ============================================================

print("\n" + "=" * 60)
print("VALEURS ATYPIQUES (IQR)")
print("=" * 60)

for col in numeriques:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR

    outliers = (
        (df[col] < borne_inf) |
        (df[col] > borne_sup)
    ).sum()

    print(f"{col:30s}: {outliers}")


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

print("\n" + "=" * 60)
print("CATEGORIES")
print("=" * 60)

for col in categoriels:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))


# ============================================================
# TAUX D'ANNULATION PAR CATEGORIE
# ============================================================

print("\n" + "=" * 60)
print("TAUX D'ANNULATION PAR CATEGORIE")
print("=" * 60)

categories_importantes = [
    "region_hotel",
    "type_destination",
    "segment_client",
    "canal_reservation",
    "moyen_transport",
    "formule_repas",
    "tarif_remboursable",
    "type_acompte",
    "client_type",
    "agent_id"
]

for col in categories_importantes:

    taux = (
        df.groupby(col)["reservation_annulee"]
        .mean()
        .sort_values(ascending=False)
    )

    print(f"\n--- {col} ---")
    print(taux)


# ============================================================
# NUMERIQUES VS ANNULATION
# ============================================================

print("\n" + "=" * 60)
print("VARIABLES NUMERIQUES VS ANNULATION")
print("=" * 60)

for col in numeriques:

    moyennes = (
        df.groupby("reservation_annulee")[col]
        .mean()
    )

    print(f"\n{col}")
    print(moyennes)


# ============================================================
# ANALYSE TEMPORELLE
# ============================================================

print("\n" + "=" * 60)
print("ANALYSE TEMPORELLE")
print("=" * 60)

taux_mois = (
    df.groupby("mois_arrivee")["reservation_annulee"]
    .mean()
)

print("\nTaux d'annulation par mois d'arrivée :")
print(taux_mois)

taux_mois.plot(
    marker="o",
    figsize=(8, 4)
)

plt.title("Taux d'annulation selon le mois d'arrivée")
plt.xlabel("Mois")
plt.ylabel("Taux d'annulation")
plt.xticks(range(1, 13))
plt.grid()


# ============================================================
# DELAI DE RESERVATION
# ============================================================

df["groupe_delai"] = pd.cut(
    df["delai_reservation_jours"],
    bins=[-1, 7, 30, 90, 180, np.inf],
    labels=["0-7", "8-30", "31-90", "91-180", "180+"]
)

taux_delai = (
    df.groupby("groupe_delai", observed=True)["reservation_annulee"]
    .mean()
)

print("\nTaux d'annulation selon le délai de réservation :")
print(taux_delai)

taux_delai.plot(
    kind="bar",
    figsize=(8, 4)
)

plt.title("Taux d'annulation selon le délai de réservation")
plt.xlabel("Délai de réservation (jours)")
plt.ylabel("Taux d'annulation")
plt.xticks(rotation=0)


# ============================================================
# HISTORIQUE CLIENT
# ============================================================

df["taux_annulation_historique"] = np.where(
    df["reservations_passees"] > 0,
    df["annulations_passees"] / df["reservations_passees"],
    0
)

print("\nTaux d'annulation historique moyen :")

print(
    df.groupby("reservation_annulee")
      ["taux_annulation_historique"]
      .mean()
)


# ============================================================
# CORRELATIONS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATIONS")
print("=" * 60)

corr = df[numeriques + ["reservation_annulee"]].corr()

print(
    corr["reservation_annulee"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f"
)

plt.title("Matrice de corrélation")
plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("EDA TERMINEE")
print("=" * 60)