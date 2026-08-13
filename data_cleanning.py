import pandas as pd


df = pd.read_csv("ressources/reservations_train.csv")

# vérification de la structure du dataset d'entrainement

# print(df.shape)
# print(df.head())
# print(df.info())

# vérifications des doublons

# print("Doublons de lignes :", df.duplicated().sum())
# print("Doublons d'ID :", df["reservation_id"].duplicated().sum())

# verification des differences de casse 

# for col in [
#     "region_hotel",
#     "type_destination",
#     "segment_client",
#     "canal_reservation",
#     "moyen_transport",
#     "formule_repas",
#     "tarif_remboursable",
#     "type_acompte",
#     "client_type"
# ]:
#     print(f"\n--- {col} ---")
#     print(df[col].value_counts(dropna=False))


