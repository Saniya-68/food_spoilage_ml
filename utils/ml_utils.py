import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "food_spoilage_pipeline.pkl")


def generate_sample_dataset(csv_path=None, n_samples=1500, random_state=42):
    np.random.seed(random_state)
    food_types = ["Vegetables", "Fruits", "Dairy", "Meat", "Grains", "Beverages"]
    storage_types = ["Room", "Fridge"]
    rows = []

    for _ in range(n_samples):
        ftype = np.random.choice(food_types)
        storage = np.random.choice(storage_types, p=[0.45, 0.55])
        temperature = np.random.normal(5, 2) if storage == "Fridge" else np.random.normal(26, 3)
        humidity = np.random.uniform(20, 85)
        purchase_age_days = np.random.randint(0, 5)

        # Spoilage baseline by food type and storage
        base_life = {
            "Vegetables": 9,
            "Fruits": 8,
            "Dairy": 5,
            "Meat": 7,
            "Grains": 20,
            "Beverages": 14,
        }[ftype]

        modifier = -1.2 * (temperature / 10.0)
        storage_bonus = 3 if storage == "Fridge" else -1
        random_noise = np.random.normal(0, 2)
        days_before_expiry = max(0, int(base_life + storage_bonus + modifier - purchase_age_days + random_noise))

        rows.append({
            "food_type": ftype,
            "storage_type": storage,
            "temperature": round(float(temperature), 1),
            "humidity": round(float(humidity), 1),
            "purchase_age_days": int(purchase_age_days),
            "days_before_expiry": days_before_expiry,
        })

    df = pd.DataFrame(rows)
    if csv_path:
        df.to_csv(csv_path, index=False)
    return df


def build_and_train(csv_path=None, model_output=MODEL_FILE):
    # Load or generate dataset
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = generate_sample_dataset(csv_path=csv_path)

    feature_cols = ["food_type", "storage_type", "temperature", "humidity", "purchase_age_days"]
    target_col = "days_before_expiry"

    X = df[feature_cols]
    y = df[target_col]

    numeric_features = ["temperature", "humidity", "purchase_age_days"]
    categorical_features = ["food_type", "storage_type"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=80, random_state=42),
    }
    results = {}

    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocess), ("estimator", model)])
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        results[name] = {
            "model": pipeline,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

    # Choose best by RMSE
    best_name = min(results, key=lambda x: results[x]["rmse"])
    best_pipeline = results[best_name]["model"]

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(best_pipeline, model_output)

    return {
        "best_model": best_name,
        "metrics": {
            k: {"mae": v["mae"], "rmse": v["rmse"], "r2": v["r2"]}
            for k, v in results.items()
        },
        "model_path": model_output,
    }


def predict_days(model_pipeline, food_type, storage_type, temperature, humidity, purchase_age_days):
    features = pd.DataFrame([
        {
            "food_type": food_type,
            "storage_type": storage_type,
            "temperature": float(temperature),
            "humidity": float(humidity),
            "purchase_age_days": int(purchase_age_days),
        }
    ])
    predicted = model_pipeline.predict(features)
    predicted_days = int(round(predicted[0]))
    return max(0, predicted_days)
