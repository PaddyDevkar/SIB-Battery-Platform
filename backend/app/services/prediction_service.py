from app.core.signal_engine.loader import load_signal
from app.core.signal_engine.cleaning import clean_signal, regenerate_cycles
from app.core.signal_engine.capacity_energy import compute_capacity_energy
from app.core.cycle_engine.cycle_summary import generate_cycle_summary
from app.core.feature_engine.feature_extractor import extract_features
from app.core.ml_engine.predictor import predict
from app.core.ml_engine.risk_classifier import classify_risk
from app.core.explain_engine.shap_explainer import explain_prediction
from app.core.prognostics_engine.lifetime_projection import (
    full_lifetime_analysis,
    project_capacity_curve
)
from app.core.explain_engine.human_explainer import generate_human_explanation
from app.core.ml_engine.health_score import compute_health_score
from app.core.prognostics_engine.soh_engine import (
    compute_energy_soh,
    compute_power_soh,
    compute_overall_soh,
    compute_remaining_useful_life,
    classify_degradation_phase
)
from app.core.explain_engine.degradation_mode import detect_degradation_mode
from app.core.prognostics_engine.failure_probability import compute_failure_probability
from app.core.ml_engine.confidence_score import compute_confidence_score


def run_prediction(file_path: str):
    """
    Full structured battery prognostics pipeline.
    """

    try:
        # ==================================================
        # 1️⃣ Load File
        # ==================================================
        df = load_signal(file_path)
        if df is None:
            return {"status": "error", "message": "Failed to load HDF5 file."}

        # ==================================================
        # 2️⃣ Signal Processing
        # ==================================================
        df = clean_signal(df)
        df = regenerate_cycles(df)
        df = compute_capacity_energy(df)

        # ==================================================
        # 3️⃣ Cycle Summary
        # ==================================================
        cycle_df = generate_cycle_summary(df)
        if cycle_df is None or len(cycle_df) < 10:
            return {"status": "error", "message": "Not enough valid cycles for prediction."}

        total_cycles = len(cycle_df)

        # ==================================================
        # 4️⃣ Feature Extraction
        # ==================================================
        features = extract_features(cycle_df, df)
        if features is None:
            return {"status": "error", "message": "Feature extraction failed."}

        # ==================================================
        # 5️⃣ ML Prediction
        # ==================================================
        prediction = predict(features)
        predicted_fade_rate = prediction["predicted_fade_rate"]
        predicted_R_mean = prediction["predicted_R_mean"]

        instability = abs(features.get("combined_instability_index", 0))
        instability_factor = 1 + instability

        risk_level = classify_risk(predicted_fade_rate)

        # ==================================================
        # 6️⃣ Lifetime Projection
        # ==================================================
        lifetime_metrics = full_lifetime_analysis(
            predicted_fade_rate,
            instability_factor=instability_factor
        )

        degradation_curve = project_capacity_curve(
            predicted_fade_rate,
            instability_factor=instability_factor
        )

        predicted_knee_cycle = degradation_curve.get("predicted_knee_cycle")

        # ==================================================
        # 7️⃣ SOH & RUL
        # ==================================================
        energy_soh = compute_energy_soh(
            current_cycle=total_cycles,
            projected_curve=degradation_curve
        )

        power_soh = compute_power_soh(predicted_R_mean)

        overall_soh = compute_overall_soh(energy_soh, power_soh)

        rul = compute_remaining_useful_life(
            current_cycles=total_cycles,
            eol_cycles=lifetime_metrics.get("estimated_end_of_life_cycles")
        )

        degradation_phase = classify_degradation_phase(
            current_cycles=total_cycles,
            eol_cycles=lifetime_metrics.get("estimated_end_of_life_cycles")
        )

        # ==================================================
        # 8️⃣ Explainability
        # ==================================================
        shap_features = explain_prediction(features, top_n=5)

        degradation_mode = detect_degradation_mode(shap_features)

        human_text = generate_human_explanation(
            fade_rate=predicted_fade_rate,
            resistance=predicted_R_mean,
            risk_level=risk_level,
            lifetime_metrics=lifetime_metrics,
            shap_features=shap_features
        )

        # ==================================================
        # 9️⃣ Advanced Metrics
        # ==================================================
        health_score = compute_health_score(
            fade_rate=predicted_fade_rate,
            resistance=predicted_R_mean,
            instability_factor=instability_factor,
            eol_cycles=lifetime_metrics.get("estimated_end_of_life_cycles")
        )

        failure_probability = compute_failure_probability(
            total_cycles,
            lifetime_metrics.get("estimated_end_of_life_cycles")
        )

        confidence_score = compute_confidence_score(
            total_cycles,
            instability_factor
        )

        # ==================================================
        # 🔟 Final Response
        # ==================================================
        return {
            "status": "success",
            "battery_analysis": {
                "predicted_fade_rate": predicted_fade_rate,
                "predicted_internal_resistance": predicted_R_mean,
                "risk_level": risk_level,
                "health_score": health_score,
                "soh_energy_percent": energy_soh,
                "soh_power_percent": power_soh,
                "soh_overall_percent": overall_soh,
                "remaining_useful_life_cycles": rul,
                "degradation_phase": degradation_phase,
                "degradation_mode": degradation_mode,
                "failure_probability_percent": failure_probability,
                "confidence_score": confidence_score,
                "predicted_knee_cycle": predicted_knee_cycle
            },
            "lifetime_prediction": lifetime_metrics,
            "degradation_curve": degradation_curve,
            "explanation": {
                "top_contributing_features": shap_features
            },
            "human_interpretation": human_text,
            "metadata": {
                "total_cycles_detected": total_cycles,
                "early_cycles_used": 30,
                "instability_factor": instability_factor
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
