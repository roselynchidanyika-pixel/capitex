"""
ML model for capital project cost overrun prediction.
Trains on synthetic Zimbabwean infrastructure data + live macro features.
Uses Gradient Boosting with SHAP-style feature importance.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import pickle
import os
from datetime import datetime

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Resolve model dir relative to this module so the layout
# (models/ subpackage vs flat deploy) doesn't matter.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_THIS_DIR, "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

SECTORS = ["roads", "buildings", "energy", "water", "telecoms", "mining", "agriculture"]
CONTRACTOR_TYPES = ["local_small", "local_large", "international"]
FUNDING_SOURCES = ["govt", "donor", "PPP", "mixed"]
PROCUREMENT_METHODS = ["open_competitive", "restricted", "direct"]
PROVINCES = [
    "Harare", "Bulawayo", "Manicaland", "Mashonaland Central",
    "Mashonaland East", "Mashonaland West", "Masvingo", "Matabeleland North",
    "Matabeleland South", "Midlands",
]

FEATURE_COLUMNS = [
    "baseline_capex", "baseline_duration_days", "complexity_score",
    "design_completeness", "procurement_delay_days", "num_change_orders",
    "sector_encoded", "contractor_encoded", "funding_encoded",
    "procurement_encoded", "province_encoded",
    "inflation_rate", "exchange_rate", "interest_rate",
    "exchange_rate_volatility", "reserves_months",
]


def generate_synthetic_training_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic project data for Zimbabwe."""
    rng = np.random.RandomState(seed)

    data = {
        "project_id": [f"ZIM-{i:04d}" for i in range(1, n_samples + 1)],
        "sector": rng.choice(SECTORS, n_samples),
        "baseline_capex": rng.lognormal(mean=np.log(5e6), sigma=1.2, size=n_samples).clip(1e5, 5e9),
        "baseline_duration_days": rng.choice([180, 270, 365, 540, 730], n_samples,
                                              p=[0.15, 0.25, 0.30, 0.20, 0.10]),
        "complexity_score": rng.randint(1, 6, n_samples),
        "design_completeness": rng.beta(3, 2, n_samples),
        "procurement_delay_days": rng.exponential(45, n_samples).clip(0, 365).astype(int),
        "num_change_orders": rng.poisson(2.5, n_samples),
        "contractor_type": rng.choice(CONTRACTOR_TYPES, n_samples, p=[0.4, 0.4, 0.2]),
        "funding_source": rng.choice(FUNDING_SOURCES, n_samples, p=[0.35, 0.30, 0.20, 0.15]),
        "procurement_method": rng.choice(PROCUREMENT_METHODS, n_samples, p=[0.5, 0.35, 0.15]),
        "province": rng.choice(PROVINCES, n_samples),
        "inflation_rate": rng.uniform(20, 120, n_samples),
        "exchange_rate": rng.lognormal(np.log(13), 0.3, n_samples),
        "interest_rate": rng.uniform(15, 80, n_samples),
        "exchange_rate_volatility": rng.uniform(0.005, 0.05, n_samples),
        "reserves_months": rng.uniform(0.3, 4.0, n_samples),
    }

    df = pd.DataFrame(data)

    # Generate realistic overrun probabilities based on features
    risk_score = (
        0.25 * (df["complexity_score"] / 5)
        + 0.20 * (df["design_completeness"] < 0.5).astype(float)
        + 0.15 * (df["procurement_delay_days"] / 180).clip(0, 1)
        + 0.10 * (df["num_change_orders"] / 8).clip(0, 1)
        + 0.10 * (df["contractor_type"] == "local_small").astype(float)
        + 0.10 * (df["inflation_rate"] / 120).clip(0, 1)
        + 0.05 * (df["exchange_rate_volatility"] / 0.05).clip(0, 1)
        + 0.05 * (1 - df["reserves_months"] / 4).clip(0, 1)
        + rng.normal(0, 0.08, n_samples)
    )

    df["overrun_20pct"] = (risk_score > 0.55).astype(int)
    df["actual_overrun_pct"] = (risk_score * 100 + rng.normal(10, 8, n_samples)).clip(-10, 200)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features to numeric."""
    df = df.copy()
    encoders = {}

    for col, categories in [
        ("sector", SECTORS), ("contractor_type", CONTRACTOR_TYPES),
        ("funding_source", FUNDING_SOURCES), ("procurement_method", PROCUREMENT_METHODS),
        ("province", PROVINCES),
    ]:
        if col in df.columns:
            mapping = {c: i for i, c in enumerate(categories)}
            df[f"{col}_encoded"] = df[col].map(mapping).fillna(-1).astype(int)

    return df


def train_model(df: pd.DataFrame) -> Tuple[object, dict]:
    """Train the overrun prediction model. Returns (model, metrics)."""
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required. pip install scikit-learn")

    df = encode_features(df)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df["overrun_20pct"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", use_label_encoder=False,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )

    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
    model.fit(X_scaled, y)

    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    train_auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0

    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    metrics = {
        "cv_roc_auc_mean": round(float(np.mean(scores)), 4),
        "cv_roc_auc_std": round(float(np.std(scores)), 4),
        "train_roc_auc": round(float(train_auc), 4),
        "n_samples": len(df),
        "positive_rate": round(float(y.mean()), 4),
        "top_features": [(f, round(float(s), 4)) for f, s in top_features[:10]],
    }

    # Save model + scaler
    with open(os.path.join(MODEL_DIR, "overrun_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols,
                      "encoders": {}}, f)

    return model, metrics


def load_model():
    """Load trained model from disk."""
    path = os.path.join(MODEL_DIR, "overrun_model.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def predict_overrun(projects_df: pd.DataFrame, macro_data: dict = None) -> pd.DataFrame:
    """Predict cost overrun probability for a list of projects."""
    model_data = load_model()
    if model_data is None:
        df = generate_synthetic_training_data()
        train_model(df)
        model_data = load_model()

    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]

    df = projects_df.copy()
    df = encode_features(df)

    if macro_data:
        df["inflation_rate"] = macro_data.get("inflation", {}).get("zimbabwe_pct", 55.0)
        df["exchange_rate"] = macro_data.get("exchange_rate", {}).get("zig_per_usd", 13.5)
        df["interest_rate"] = macro_data.get("interest_rates", {}).get("zim_policy_pct", 35.0)
        df["exchange_rate_volatility"] = 0.02
        df["reserves_months"] = macro_data.get("reserves", {}).get("total_usd_bn", 0.5) * 2

    for col in ["exchange_rate_volatility", "reserves_months"]:
        if col not in df.columns:
            df[col] = 0.02 if "volatility" in col else 1.0

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]

    df["overrun_probability"] = probabilities
    df["risk_category"] = pd.cut(
        probabilities,
        bins=[0, 0.30, 0.60, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    # Feature contribution (simplified SHAP-like), stored as JSON string for
    # Arrow-compatible serialization in Streamlit dataframes.
    # Align importances to the actual available feature columns.
    import json as _json
    importances = model.feature_importances_
    driver_cols = available  # only columns actually present in X

    def _format_drivers(idx):
        entries = [
            (driver_cols[i], round(float(importances[i] * X.iloc[idx, i]), 4))
            for i in range(len(driver_cols))
        ]
        entries.sort(key=lambda x: abs(x[1]), reverse=True)
        return _json.dumps(entries[:5])

    df["top_risk_drivers"] = [_format_drivers(idx) for idx in range(len(df))]

    return df


def get_model_info() -> dict:
    """Return model metadata and performance metrics."""
    model_data = load_model()
    if model_data is None:
        return {"status": "not_trained"}

    model = model_data["model"]
    return {
        "status": "trained",
        "model_type": type(model).__name__,
        "n_features": len(model_data["feature_cols"]),
        "feature_cols": model_data["feature_cols"],
    }


if __name__ == "__main__":
    print("Generating synthetic data and training model...")
    df = generate_synthetic_training_data(500)
    model, metrics = train_model(df)
    print(f"Model: {type(model).__name__}")
    print(f"CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f}")
    print(f"Top features: {metrics['top_features'][:5]}")

    print("\nPredicting on sample projects...")
    sample = df.head(10)
    predictions = predict_overrun(sample)
    print(predictions[["project_id", "sector", "overrun_probability", "risk_category"]].to_string(index=False))
