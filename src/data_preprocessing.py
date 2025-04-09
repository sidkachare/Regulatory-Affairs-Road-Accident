import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath: str):
    df = pd.read_csv(filepath)

    df.dropna(subset=["Count"], inplace=True)
    df["Count"] = df["Count"].astype(int)

    le_city = LabelEncoder()
    le_cause = LabelEncoder()
    le_subcause = LabelEncoder()
    le_outcome = LabelEncoder()

    df["City_encoded"] = le_city.fit_transform(df["Million Plus Cities"])
    df["Cause_encoded"] = le_cause.fit_transform(df["Cause category"])
    df["Subcause_encoded"] = le_subcause.fit_transform(df["Cause Subcategory"])
    df["Outcome_encoded"] = le_outcome.fit_transform(df["Outcome of Incident"])

    X = df[["City_encoded", "Cause_encoded", "Subcause_encoded", "Outcome_encoded"]]
    y = df["Count"]

    encoders = {
        "city": le_city,
        "cause": le_cause,
        "subcause": le_subcause,
        "outcome": le_outcome
    }

    return X, y, encoders
