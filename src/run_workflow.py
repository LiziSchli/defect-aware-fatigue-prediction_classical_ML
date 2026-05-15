import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = PROJECT_ROOT / "data" / "simulated_defect_fatigue_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.drop_duplicates(subset="specimen_id").reset_index(drop=True)

    numeric_columns = [
        "defect_size_um",
        "defect_depth_um",
        "sphericity",
        "compactness",
        "aspect_ratio",
        "local_stress_mpa",
        "stress_ratio",
        "relative_density_pct",
        "fatigue_life_cycles",
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean["defect_size_um"] = clean["defect_size_um"].clip(lower=10.0, upper=1000.0)
    clean["defect_depth_um"] = clean["defect_depth_um"].clip(lower=10.0, upper=5000.0)
    clean["sphericity"] = clean["sphericity"].clip(lower=0.0, upper=1.0)
    clean["compactness"] = clean["compactness"].clip(lower=0.0, upper=1.0)
    clean["aspect_ratio"] = clean["aspect_ratio"].clip(lower=0.3, upper=10.0)
    clean["local_stress_mpa"] = clean["local_stress_mpa"].clip(lower=50.0, upper=500.0)
    clean["stress_ratio"] = clean["stress_ratio"].clip(lower=-1.0, upper=1.0)
    clean["relative_density_pct"] = clean["relative_density_pct"].clip(lower=90.0, upper=100.0)
    clean["fatigue_life_cycles"] = clean["fatigue_life_cycles"].clip(lower=1.0)
    return clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    size_filled = engineered["defect_size_um"].fillna(engineered["defect_size_um"].median()).clip(lower=1.0)
    depth_filled = engineered["defect_depth_um"].fillna(engineered["defect_depth_um"].median()).clip(lower=1.0)
    sphericity_filled = engineered["sphericity"].fillna(engineered["sphericity"].median())
    compactness_filled = engineered["compactness"].fillna(engineered["compactness"].median())
    aspect_ratio_filled = engineered["aspect_ratio"].fillna(engineered["aspect_ratio"].median()).clip(lower=0.5)
    stress_filled = engineered["local_stress_mpa"].fillna(engineered["local_stress_mpa"].median()).clip(lower=1.0)
    density_filled = engineered["relative_density_pct"].fillna(engineered["relative_density_pct"].median())

    engineered["surface_proximity_ratio"] = depth_filled / size_filled
    engineered["surface_defect_flag"] = (engineered["surface_proximity_ratio"] < 1.0).astype(int)
    engineered["stress_intensity_proxy"] = stress_filled * np.sqrt(size_filled / 1000.0)
    engineered["morphology_risk_score"] = (
        (1.0 - sphericity_filled)
        + (1.0 - compactness_filled)
        + np.log1p(np.maximum(aspect_ratio_filled - 1.0, 0.0))
    )
    engineered["density_defect_interaction"] = size_filled * (100.0 - density_filled)
    engineered["log10_fatigue_life"] = np.log10(engineered["fatigue_life_cycles"].clip(lower=1.0))
    return engineered


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model) -> Pipeline:
    numeric_features = [
        "defect_size_um",
        "defect_depth_um",
        "sphericity",
        "compactness",
        "aspect_ratio",
        "local_stress_mpa",
        "stress_ratio",
        "relative_density_pct",
        "surface_proximity_ratio",
        "surface_defect_flag",
        "stress_intensity_proxy",
        "morphology_risk_score",
        "density_defect_interaction",
    ]
    categorical_features = [
        "material_family",
        "build_orientation",
        "heat_treatment",
        "defect_type",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_models(x: pd.DataFrame, y: pd.Series, cv: KFold) -> pd.DataFrame:
    model_zoo = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    for model_name, model in model_zoo.items():
        pipeline = build_pipeline(model)
        scores = cross_validate(
            pipeline,
            x,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
        mae_values = -scores["test_mae"]
        rmse_values = np.sqrt(-scores["test_mse"])
        rows.append(
            {
                "model": model_name,
                "cv_mae_mean": float(mae_values.mean()),
                "cv_mae_std": float(mae_values.std(ddof=0)),
                "cv_rmse_mean": float(rmse_values.mean()),
                "cv_r2_mean": float(scores["test_r2"].mean()),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)
    return metrics


def fit_and_predict_best_model(
    x: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
    best_model_name: str,
) -> tuple[Pipeline, pd.DataFrame]:
    model_lookup = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    pipeline = build_pipeline(model_lookup[best_model_name])
    y_pred_cv = cross_val_predict(pipeline, x, y, cv=cv)
    pipeline.fit(x, y)

    prediction_table = pd.DataFrame(
        {
            "actual_log10_life": y,
            "predicted_log10_life": y_pred_cv,
            "actual_cycles": np.power(10.0, y),
            "predicted_cycles": np.power(10.0, y_pred_cv),
        }
    )
    return pipeline, prediction_table


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    else:
        raise ValueError("Selected model does not expose feature importance.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(metrics["model"], metrics["cv_mae_mean"], color="#2E6F95")
    axes[0].set_title("Cross-Validated MAE")
    axes[0].set_ylabel("MAE on log10 fatigue life")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metrics["model"], metrics["cv_r2_mean"], color="#D97D54")
    axes[1].set_title("Cross-Validated R2")
    axes[1].set_ylabel("R2")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(prediction_table: pd.DataFrame, output_path: Path) -> None:
    actual = prediction_table["actual_log10_life"].to_numpy()
    predicted = prediction_table["predicted_log10_life"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(actual, predicted, alpha=0.75, color="#33658A", edgecolor="white", linewidth=0.5)

    bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(bounds, bounds, linestyle="--", color="#BC4B51", linewidth=1.5)
    ax.set_title("Cross-Validated Predictions vs Actual")
    ax.set_xlabel("Actual log10 fatigue life")
    ax.set_ylabel("Predicted log10 fatigue life")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    top = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.barh(top["feature"], top["importance"], color="#7A9E7E")
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    feature_columns = [
        "material_family",
        "build_orientation",
        "heat_treatment",
        "defect_type",
        "defect_size_um",
        "defect_depth_um",
        "sphericity",
        "compactness",
        "aspect_ratio",
        "local_stress_mpa",
        "stress_ratio",
        "relative_density_pct",
        "surface_proximity_ratio",
        "surface_defect_flag",
        "stress_intensity_proxy",
        "morphology_risk_score",
        "density_defect_interaction",
    ]
    target_column = "log10_fatigue_life"

    x = df[feature_columns]
    y = df[target_column]
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics = evaluate_models(x, y, cv=cv)
    metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)

    best_model_name = metrics.iloc[0]["model"]
    best_pipeline, prediction_table = fit_and_predict_best_model(x, y, cv=cv, best_model_name=best_model_name)

    prediction_table["absolute_error"] = np.abs(
        prediction_table["predicted_log10_life"] - prediction_table["actual_log10_life"]
    )
    prediction_table.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False)

    summary = {
        "best_model": best_model_name,
        "mae": float(mean_absolute_error(prediction_table["actual_log10_life"], prediction_table["predicted_log10_life"])),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    prediction_table["actual_log10_life"],
                    prediction_table["predicted_log10_life"],
                )
            )
        ),
        "r2": float(r2_score(prediction_table["actual_log10_life"], prediction_table["predicted_log10_life"])),
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "best_model_summary.csv", index=False)

    importance_df = extract_feature_importance(best_pipeline)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plot_model_comparison(metrics, OUTPUT_DIR / "model_comparison.png")
    plot_predictions(prediction_table, OUTPUT_DIR / "predictions_vs_actual.png")
    plot_feature_importance(importance_df, OUTPUT_DIR / "top_feature_importance.png")

    print(f"Best model: {best_model_name}")
    print(f"Wrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
