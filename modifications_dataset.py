import pandas as pd
df = pd.read_csv("../ressources/reservations_train.csv")
df["agent_id"] = df["agent_id"].fillna("DIRECT")
print(df["agent_id"].value_counts())
print(df["agent_id"].isna().sum())

print(
    df.groupby("agent_id")["reservation_annulee"]
      .mean()
      .sort_values(ascending=False)
)