from __future__ import annotations

import json
import os
import warnings
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "initial_data.csv"
ARTIFACT_DIR = Path(os.getenv("GROCYGENIE_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "consumption_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

REGIONS = ("urban", "rural")
SEASONS = ("winter", "spring", "summer", "autumn")
EVENTS = ("normal", "fasting", "guests", "sickness", "travel", "meal_off")

BASE_CONSUMPTION = {
    "rice": {"adult_male": 0.3, "adult_female": 0.25, "child": 0.15},
    "milk": {"adult_male": 0.2, "adult_female": 0.18, "child": 0.3},
    "potato": {"adult_male": 0.25, "adult_female": 0.2, "child": 0.15},
    "onion": {"adult_male": 0.1, "adult_female": 0.1, "child": 0.05},
    "lentils": {"adult_male": 0.15, "adult_female": 0.12, "child": 0.08},
    "flour": {"adult_male": 0.2, "adult_female": 0.18, "child": 0.1},
    "tea": {"adult_male": 0.01, "adult_female": 0.01, "child": 0.005},
    "coffee": {"adult_male": 0.02, "adult_female": 0.02, "child": 0.0},
    "almond": {"adult_male": 0.03, "adult_female": 0.03, "child": 0.01},
    "sugar": {"adult_male": 0.05, "adult_female": 0.05, "child": 0.03},
}

UNIT_CONVERSION_FACTORS = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "l": 1.0,
    "litre": 1.0,
    "litres": 1.0,
    "lt": 1.0,
    "ml": 0.001,
    "millilitre": 0.001,
    "millilitres": 0.001,
    "pcs": 1.0,
    "piece": 1.0,
    "pieces": 1.0,
}

CATEGORICAL_FEATURES = ["product", "region", "season", "event"]
NUMERIC_FEATURES = [
    "adult_male",
    "adult_female",
    "child",
    "family_size",
    "base_consumption",
    "month",
    "day_of_week",
]
TARGET = "consumption"

_model: Pipeline | None = None
_metadata: dict[str, Any] = {}


@dataclass(frozen=True)
class PredictionResult:
    daily_consumption: float
    days_to_finish: float
    predicted_finish_date: date


def get_season(dt_obj: date | datetime) -> str:
    month = dt_obj.month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def normalize_text(value: str | None, default: str) -> str:
    value = (value or default).strip().lower()
    return value or default


def validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    normalized = normalize_text(value, allowed[0])
    if normalized not in allowed:
        allowed_values = ", ".join(allowed)
        raise ValueError(f"Invalid {name} '{value}'. Allowed values: {allowed_values}.")
    return normalized


def normalize_unit(unit: str | None) -> str:
    normalized = normalize_text(unit, "kg")
    if normalized not in UNIT_CONVERSION_FACTORS:
        allowed_values = ", ".join(sorted(UNIT_CONVERSION_FACTORS))
        raise ValueError(f"Invalid unit '{unit}'. Allowed values: {allowed_values}.")
    return normalized


def quantity_to_base_unit(quantity: float, unit: str | None) -> float:
    normalized_unit = normalize_unit(unit)
    return float(quantity) * UNIT_CONVERSION_FACTORS[normalized_unit]


def family_to_dict(family: Any) -> dict[str, int]:
    if family is None:
        return {"adult_male": 1, "adult_female": 1, "child": 0}
    if isinstance(family, Mapping):
        data = dict(family)
    elif hasattr(family, "model_dump"):
        data = family.model_dump()
    elif hasattr(family, "dict"):
        data = family.dict()
    else:
        raise ValueError("Family data must include adult_male, adult_female, and child.")
    return {
        "adult_male": max(int(data.get("adult_male", 0)), 0),
        "adult_female": max(int(data.get("adult_female", 0)), 0),
        "child": max(int(data.get("child", 0)), 0),
    }


def calculate_base_consumption(
    family: Mapping[str, int],
    region: str,
    season: str,
    event: str,
    product: str,
) -> float:
    base = BASE_CONSUMPTION.get(
        product,
        {"adult_male": 0.08, "adult_female": 0.07, "child": 0.04},
    )
    total = sum(base.get(member, 0.0) * family.get(member, 0) for member in family)
    region_multiplier = {"urban": 1.0, "rural": 1.1}.get(region, 1.0)
    season_multiplier = {
        "winter": 1.1,
        "spring": 1.0,
        "summer": 0.9,
        "autumn": 1.0,
    }.get(season, 1.0)
    event_multiplier = {
        "normal": 1.0,
        "fasting": 0.7,
        "guests": 1.3,
        "sickness": 0.5,
        "travel": 0.3,
        "meal_off": 0.2,
    }.get(event, 1.0)
    return max(total * region_multiplier * season_multiplier * event_multiplier, 0.001)


def load_training_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    df = pd.read_csv(data_path)
    required = set(CATEGORICAL_FEATURES + ["adult_male", "adult_female", "child", "date", TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")
    return clean_training_data(df)


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET, "product", "region", "season", "event"])

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str).str.strip().str.lower()

    for column in ["adult_male", "adult_female", "child", TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["adult_male", "adult_female", "child", TARGET])
    df = df[df[TARGET] > 0]
    df = df[df["region"].isin(REGIONS)]
    df = df[df["season"].isin(SEASONS)]
    df = df[df["event"].isin(EVENTS)]
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family_size"] = df["adult_male"] + df["adult_female"] + df["child"]
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["base_consumption"] = [
        calculate_base_consumption(
            {
                "adult_male": int(row.adult_male),
                "adult_female": int(row.adult_female),
                "child": int(row.child),
            },
            row.region,
            row.season,
            row.event,
            row.product,
        )
        for row in df.itertuples(index=False)
    ]
    return df


def build_pipeline(model_type: str = "hist_gradient_boosting") -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", encoder, CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    if model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=350,
            l2_regularization=0.01,
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def train_model(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    model_type: str = "hist_gradient_boosting",
) -> dict[str, Any]:
    global _model, _metadata

    df = load_training_data(data_path)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["product"],
    )

    pipeline = build_pipeline(model_type=model_type)
    pipeline.fit(train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train_df[TARGET])

    metrics = evaluate_pipeline(pipeline, test_df)
    metadata = {
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "products": sorted(df["product"].unique().tolist()),
        "features": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_path)
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _model = pipeline
    _metadata = metadata
    return metadata


def evaluate_pipeline(pipeline: Pipeline, df: pd.DataFrame) -> dict[str, Any]:
    features = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    actual = df[TARGET].to_numpy()
    predicted = np.clip(pipeline.predict(features), 0.001, None)

    absolute_error = np.abs(actual - predicted)
    percentage_error = absolute_error / np.maximum(actual, 0.001) * 100

    by_product: dict[str, dict[str, float]] = {}
    for product, group in df.assign(predicted=predicted).groupby("product"):
        group_actual = group[TARGET].to_numpy()
        group_predicted = group["predicted"].to_numpy()
        by_product[product] = {
            "mae": round(float(mean_absolute_error(group_actual, group_predicted)), 4),
            "mape": round(
                float(np.mean(np.abs(group_actual - group_predicted) / np.maximum(group_actual, 0.001)) * 100),
                2,
            ),
        }

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(rmse, 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
        "mape": round(float(np.mean(percentage_error)), 2),
        "median_absolute_error": round(float(np.median(absolute_error)), 4),
        "by_product": by_product,
    }


def load_model() -> Pipeline:
    global _model, _metadata

    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        train_model()

    artifact = joblib.load(MODEL_PATH)
    _model = artifact["pipeline"]
    _metadata = artifact.get("metadata", {})
    return _model


def get_model_metadata() -> dict[str, Any]:
    load_model()
    return dict(_metadata)


def get_supabase_client():
    from supabase_client import get_supabase

    return get_supabase()


def get_user_details(user_id: str) -> dict[str, Any] | None:
    response = (
        get_supabase_client()
        .table("users")
        .select("*")
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    return response.data


def build_prediction_frame(
    product_name: str,
    family: Mapping[str, int],
    region: str,
    season: str,
    event: str,
    purchase_date: date,
) -> pd.DataFrame:
    product = normalize_text(product_name, "unknown")
    region = validate_choice("region", region, REGIONS)
    season = validate_choice("season", season, SEASONS)
    event = validate_choice("event", event, EVENTS)

    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(purchase_date),
                "product": product,
                "region": region,
                "season": season,
                "event": event,
                "adult_male": family["adult_male"],
                "adult_female": family["adult_female"],
                "child": family["child"],
                TARGET: 1.0,
            }
        ]
    )
    return add_features(frame)[CATEGORICAL_FEATURES + NUMERIC_FEATURES]


def predict_depletion(
    product_name: str,
    quantity: float,
    unit: str,
    purchase_date: date,
    family: Mapping[str, int],
    region: str,
    season: str,
    event: str,
) -> PredictionResult:
    if quantity <= 0:
        raise ValueError("Quantity must be greater than 0.")

    model = load_model()
    quantity_in_base_unit = quantity_to_base_unit(quantity, unit)
    features = build_prediction_frame(product_name, family, region, season, event, purchase_date)
    daily_consumption = max(float(model.predict(features)[0]), 0.001)
    days_to_finish = quantity_in_base_unit / daily_consumption

    return PredictionResult(
        daily_consumption=daily_consumption,
        days_to_finish=days_to_finish,
        predicted_finish_date=purchase_date + timedelta(days=days_to_finish),
    )


def predict_and_record_stock(input_data: Any) -> dict[str, Any]:
    user_details = get_user_details(input_data.user_id)
    if not user_details:
        raise ValueError(f"User with ID {input_data.user_id} not found.")

    family = family_to_dict(input_data.family) if input_data.family else family_to_dict(user_details)
    region = input_data.region or user_details.get("region", "urban")
    season = input_data.season or get_season(input_data.purchase_date)
    event = input_data.event or "normal"

    prediction = predict_depletion(
        product_name=input_data.product_name,
        quantity=input_data.quantity,
        unit=input_data.unit,
        purchase_date=input_data.purchase_date,
        family=family,
        region=region,
        season=season,
        event=event,
    )

    stock_entry = {
        "user_id": input_data.user_id,
        "product_name": normalize_text(input_data.product_name, "unknown"),
        "unit": normalize_unit(input_data.unit),
        "quantity": float(input_data.quantity),
        "purchase_date": input_data.purchase_date.isoformat(),
        "household_events": validate_choice("event", event, EVENTS),
        "season": validate_choice("season", season, SEASONS),
        "predicted_finish_date": prediction.predicted_finish_date.isoformat(),
    }

    response = get_supabase_client().table("user_stocks").insert(stock_entry).execute()
    if not response.data:
        raise RuntimeError("Failed to insert stock record.")

    return {
        "stock_id": response.data[0]["stock_id"],
        "predicted_finish_date": prediction.predicted_finish_date.isoformat(),
        "daily_consumption": prediction.daily_consumption,
        "days_to_finish": prediction.days_to_finish,
    }


def recalculate_depletion(input_data: Any) -> str:
    prediction = predict_depletion(
        product_name=input_data.product_name,
        quantity=input_data.quantity,
        unit=input_data.unit,
        purchase_date=date.today(),
        family=family_to_dict(input_data.family),
        region=input_data.region,
        season=input_data.season,
        event=input_data.event,
    )
    return prediction.predicted_finish_date.isoformat()


def record_actual_finish_date(stock_id: uuid.UUID, actual_finish_date: date) -> bool:
    response = (
        get_supabase_client()
        .table("user_stocks")
        .update({"actual_finish_date": actual_finish_date.isoformat(), "is_verified": True})
        .eq("stock_id", str(stock_id))
        .execute()
    )
    return bool(response.data)


def retrain_model_with_feedback(user_id: str) -> dict[str, Any]:
    response = (
        get_supabase_client()
        .table("user_stocks")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_verified", True)
        .execute()
    )
    if not response.data:
        return {"success": False, "message": "No verified feedback found to retrain the model."}

    feedback_rows = []
    user_details = get_user_details(user_id) or {}
    family = family_to_dict(user_details)
    region = validate_choice("region", user_details.get("region", "urban"), REGIONS)

    for record in response.data:
        try:
            purchase_date = datetime.fromisoformat(str(record["purchase_date"]).replace("Z", "+00:00")).date()
            actual_finish_date = datetime.fromisoformat(
                str(record["actual_finish_date"]).replace("Z", "+00:00")
            ).date()
            duration_days = max((actual_finish_date - purchase_date).days, 1)
            daily_consumption = quantity_to_base_unit(float(record["quantity"]), record.get("unit")) / duration_days
            season = validate_choice("season", record.get("season") or get_season(purchase_date), SEASONS)
            event = validate_choice("event", record.get("household_events") or "normal", EVENTS)
        except (KeyError, TypeError, ValueError):
            continue

        for offset in range(duration_days):
            day = purchase_date + timedelta(days=offset)
            feedback_rows.append(
                {
                    "date": day.isoformat(),
                    "product": normalize_text(record.get("product_name"), "unknown"),
                    "region": region,
                    "season": season,
                    "event": event,
                    "adult_male": family["adult_male"],
                    "adult_female": family["adult_female"],
                    "child": family["child"],
                    TARGET: daily_consumption,
                }
            )

    if not feedback_rows:
        return {"success": False, "message": "No valid feedback rows could be processed."}

    existing = pd.read_csv(DATA_PATH)
    combined = pd.concat([existing, pd.DataFrame(feedback_rows)], ignore_index=True)
    train_model_from_dataframe(combined)

    stock_ids = [str(record["stock_id"]) for record in response.data if record.get("stock_id")]
    if stock_ids:
        get_supabase_client().table("user_stocks").update({"is_verified": False}).in_("stock_id", stock_ids).execute()

    return {"success": True, "message": f"Model retrained with {len(feedback_rows)} feedback rows."}


def train_model_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = ARTIFACT_DIR / "combined_training_data.csv"
    df.to_csv(temp_path, index=False)
    return train_model(data_path=temp_path)
