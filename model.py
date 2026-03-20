# ml/model.py
# Trains a Random Forest classifier on sensor data.
# Also includes a rule-based baseline for comparison.
#
# Usage:
#   python model.py train          -- train and save model
#   python model.py evaluate       -- load saved model and evaluate

import sys
import pickle
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

from data_loader import load_data, get_xy, engineer_features, FEATURE_COLS

MODEL_PATH = Path("models/fault_model.pkl")


# ── Rule-based baseline (no ML, pure thresholds) ──────────────────────────
def rule_based_predict(df) -> np.ndarray:
    """
    Hard-coded thresholds from domain knowledge.
    Useful as a sanity check and recruiter talking point.
    """
    import numpy as np
    flags = np.zeros(len(df), dtype=int)
    flags |= (df["temperature_c"]   > 110).astype(int)
    flags |= (df["vibration_mms"]   > 4.5).astype(int)
    flags |= (df["pressure_bar"]    < 2.0).astype(int)
    flags |= (df["current_a"]       > 25.0).astype(int)
    flags |= (df["output_rate_uph"] < 40.0).astype(int)
    return flags


# ── ML pipeline ───────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",  # handles 8% fault class imbalance
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


def train(csv_path: str = "data/sensor_data.csv") -> Pipeline:
    df       = load_data(csv_path)
    X, y, df_feat = get_xy(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[model] Training Random Forest (200 trees) ...")
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]

    print("\n===== Evaluation on 20% hold-out =====")
    print(classification_report(y_te, y_pred, target_names=["Normal", "Fault"]))
    print(f"ROC-AUC : {roc_auc_score(y_te, y_prob):.4f}")
    print(f"F1 Score: {f1_score(y_te, y_pred):.4f}")

    cm = confusion_matrix(y_te, y_pred)
    print(f"\nConfusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    # Feature importance
    rf_clf = pipe.named_steps["clf"]
    ranked = sorted(zip(FEATURE_COLS, rf_clf.feature_importances_), key=lambda x: -x[1])
    print("\nTop feature importances:")
    for feat, score in ranked[:6]:
        print(f"  {feat:<25} {score:.4f}")

    # Rule-based baseline comparison
    rule_pred = rule_based_predict(df_feat)
    print(f"\n[baseline] Rule-based F1 on full set: {f1_score(y, rule_pred):.4f}")
    print(f"[ml]       ML F1 on hold-out:          {f1_score(y_te, y_pred):.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\n[model] Saved -> {MODEL_PATH}")

    return pipe


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}.\nRun: python model.py train"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_live(readings: dict) -> dict:
    """
    Single-reading prediction for the dashboard and Flask API.
    Pass a dict with the 6 raw sensor keys; engineered features are computed here.
    """
    pipe = load_model()
    r    = readings.copy()

    # Compute engineered features
    r.setdefault("temp_vib_ratio",   r["temperature_c"]   / (r["vibration_mms"]   + 1e-6))
    r.setdefault("power_approx",     r["current_a"]       *  r["pressure_bar"])
    r.setdefault("efficiency_index", r["output_rate_uph"] / (r["cycle_time_s"]    + 1e-6))
    r.setdefault("temp_zscore", 0.0)
    r.setdefault("vib_zscore",  0.0)

    X    = np.array([[r[c] for c in FEATURE_COLS]], dtype=np.float32)
    prob = float(pipe.predict_proba(X)[0][1])

    # Rule-based safety net: z-scores need history; rules catch obvious exceedances
    rule_fault = (
        r["temperature_c"]   > 110 or
        r["vibration_mms"]   > 4.5 or
        r["pressure_bar"]    < 2.0 or
        r["current_a"]       > 25.0 or
        r["output_rate_uph"] < 40.0
    )
    if rule_fault and prob < 0.5:
        prob = max(prob, 0.65)   # push above decision threshold

    pred = int(prob >= 0.5)

    # Map to human-readable fault type using threshold rules
    fault_type = "None"
    if pred:
        if   r["temperature_c"]   > 110: fault_type = "Overheating"
        elif r["vibration_mms"]   > 4.5: fault_type = "Bearing Wear"
        elif r["pressure_bar"]    < 2.0: fault_type = "Pressure Drop"
        elif r["current_a"]       > 25.0: fault_type = "Electrical Surge"
        else:                             fault_type = "Unknown Anomaly"

    return {
        "fault_predicted":   pred,
        "fault_probability": round(prob, 4),
        "fault_type":        fault_type,
        "risk_level":        "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
    }


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "evaluate":
        model = load_model()
        df    = load_data()
        X, y, df_feat = get_xy(df)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        print(classification_report(y, y_pred, target_names=["Normal","Fault"]))
        print(f"ROC-AUC: {roc_auc_score(y, y_prob):.4f}")
    else:
        print("Usage: python model.py [train|evaluate]")
