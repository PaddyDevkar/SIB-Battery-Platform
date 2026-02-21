import shap
import pandas as pd

from app.core.ml_engine.model_loader import get_model_objects


def explain_prediction(features: dict, top_n: int = 5):
    """
    Returns top contributing SHAP features.
    Fully aligned with trained feature schema.
    """

    # Load model objects
    model, scaler, feature_columns = get_model_objects()

    # Use fade model (first estimator)
    rf_model = model.estimators_[0]

    # Create explainer
    explainer = shap.TreeExplainer(rf_model)

    # Convert to DataFrame
    X = pd.DataFrame([features])

    # 🔥 STRICT ALIGNMENT (same as predictor)
    X = X.reindex(columns=feature_columns, fill_value=0)

    # Scale
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_columns
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(X_scaled)

    # Pair features with values
    feature_impacts = list(zip(feature_columns, shap_values[0]))

    # Sort by absolute importance
    feature_impacts_sorted = sorted(
        feature_impacts,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return [
        {"feature": name, "impact": float(value)}
        for name, value in feature_impacts_sorted[:top_n]
    ]
