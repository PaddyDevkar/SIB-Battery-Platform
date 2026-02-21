import pandas as pd
from app.core.ml_engine.model_loader import get_model_objects


def predict(features: dict):

    model, scaler, feature_columns = get_model_objects()

    # Create DataFrame
    X = pd.DataFrame([features])

    # Strict align
    X = X.reindex(columns=feature_columns, fill_value=0)

    # 🔥 IMPORTANT: keep as DataFrame
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_columns
    )

    preds = model.predict(X_scaled)

    return {
        "predicted_fade_rate": float(preds[0][0]),
        "predicted_R_mean": float(preds[0][1])
    }
