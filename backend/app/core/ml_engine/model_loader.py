import os
import joblib

# -------------------------------------------------
# Define model paths
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "sib_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "sib_scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "sib_features.pkl")


# -------------------------------------------------
# Load model artifacts
# -------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    print("✅ SIB model loaded successfully.")

except Exception as e:
    print("❌ Error loading model artifacts:", e)
    model = None
    scaler = None
    feature_columns = None


# -------------------------------------------------
# Getter function (safe access)
# -------------------------------------------------

def get_model_objects():
    """
    Returns:
        model
        scaler
        feature_columns
    """
    return model, scaler, feature_columns
