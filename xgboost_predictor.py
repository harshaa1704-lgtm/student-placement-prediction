import joblib
import numpy as np
import pandas as pd


# ==========================================================
# STEP 5 — XGBOOST INTEGRATION
# ==========================================================

MODEL_PATH = r"D:\Mini project\Student_Placement_Project\models\best_model.pkl"
SCALER_PATH = r"D:\Mini project\Student_Placement_Project\models\scaler.pkl"
FEATURE_PATH = r"D:\Mini project\Student_Placement_Project\models\feature_names.pkl"


# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURE_PATH)


# ----------------------------------------------------------
# XGBOOST PREDICTION FUNCTION
# ----------------------------------------------------------

def predict_placement(features):
    """
    Predict student placement using trained XGBoost model.

    Parameters
    ----------
    features : dict
        Dictionary containing model features.

    Returns
    -------
    dict
        Placement prediction and probability.
    """

    try:

        # --------------------------------------------------
        # Create input dataframe
        # --------------------------------------------------

        input_data = {}

        for feature in feature_names:

            input_data[feature] = features.get(
                feature,
                0
            )

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # --------------------------------------------------
        # Convert to numeric
        # --------------------------------------------------

        input_df = input_df.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)

        # --------------------------------------------------
        # Scaling
        # --------------------------------------------------

        scaled_data = scaler.transform(
            input_df
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        prediction = model.predict(
            scaled_data
        )[0]

        # --------------------------------------------------
        # Probability
        # --------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_data
            )[0]

            probability = float(
                np.max(probabilities)
            ) * 100

        # --------------------------------------------------
        # Placement result
        # --------------------------------------------------

        if int(prediction) == 1:

            result = "Placed"

        else:

            result = "Not Placed"

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {
            "success": True,
            "prediction": int(prediction),
            "result": result,
            "probability": round(
                probability, 2
            ) if probability is not None else None
        }

    except Exception as e:

        return {
            "success": False,
            "prediction": None,
            "result": "Prediction Failed",
            "probability": None,
            "error": str(e)
        }