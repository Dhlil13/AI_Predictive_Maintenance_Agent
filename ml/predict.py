import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)

def predict_failure(
    air_temp,
    process_temp,
    rpm,
    torque,
    wear
):

    X = pd.DataFrame(
        {
            "Air temperature [K]":[air_temp],
            "Process temperature [K]":[process_temp],
            "Rotational speed [rpm]":[rpm],
            "Torque [Nm]":[torque],
            "Tool wear [min]":[wear]
        }
    )

    X_scaled = scaler.transform(X)

    prediction = model.predict(
        X_scaled
    )[0]

    probability = model.predict_proba(
        X_scaled
    )[0][1]

    return prediction, probability