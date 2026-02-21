import shap
from app.core.ml_engine.model_loader import get_model_objects

def get_explainer():
    model, _, feature_columns = get_model_objects()
    rf_model = model.estimators_[0]
    explainer = shap.TreeExplainer(rf_model)
    return explainer, feature_columns



def get_explainer():
    return explainer, feature_columns
