from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore
import json
import joblib
import os
import pandas as pd

service_account = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account)
firebase_admin.initialize_app(cred)
db = firestore.client()

rows = []

for survey_doc in db.collection("surveys").stream():
    responses = (
        survey_doc.reference
        .collection("responses")
        .stream()
    )

    for response_doc in responses:
        row = response_doc.to_dict()
        rows.append(row)

responses_df = pd.DataFrame(rows)

features = [
    "gender4", "CC22_320d", "CC22_320g", "CC22_363",
    "presvote20post", "newsint", "birthyr"
]

missing = set(features) - set(responses_df.columns)
if missing:
    raise ValueError(f"Missing model fields: {sorted(missing)}")

models = joblib.load("election_models.joblib")
X = responses_df[features]
projected_shares = {}

for race, model in models.items():
    shares = model.predict_proba(X).mean(axis=0)
    projected_shares[race] = pd.Series(shares, index=model.classes_)

projected_shares_df = pd.DataFrame(projected_shares)
projected_shares_df.index.name = "candidate"
projected_shares_df = projected_shares_df.reset_index()
projected_shares_df["candidate"] = projected_shares_df["candidate"].astype("Int64")

prediction_ref = db.collection("prediction")

for race in projected_shares_df.columns.drop("candidate"):
    shares = {
        str(row["candidate"]): float(row[race])
        for _, row in projected_shares_df.iterrows()
    }

    prediction_ref.document(race).set({
        "race": race,
        "shares": shares,
        "share_unit": "proportion",
        "updated_at": datetime.now(timezone.utc),
    })