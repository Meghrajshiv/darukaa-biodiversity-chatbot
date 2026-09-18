import pandas as pd

def load_reference_table(path="data/reference_table.csv"):
    return pd.read_csv(path)

def query_reference_table(df, metric, value):
    """Find matching rows for a given metric and value."""
    matches = df[df["metric"] == metric]
    results = []
    for _, row in matches.iterrows():
        if row["condition"] == "below" and value < row["threshold"]:
            results.append(row)
        elif row["condition"] == "above" and value > row["threshold"]:
            results.append(row)
        elif row["condition"] == "equals" and str(value) == str(row["threshold"]):
            results.append(row)
   