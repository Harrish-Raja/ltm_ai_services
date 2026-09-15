import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

unwanted_cols = ["order_id", "dispatch_date", "origin_warehouse_code", "destination_city", "courier_partner"]

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop_duplicates().reset_index(drop=True)
    n_cols = ["distance_km", "package_weight_kg"]

    for i in n_cols:
        df[i] = df[i].fillna(df[i].median())

    df['shipping_speed'] = (df['shipping_speed'].astype(str).str.lower().str.strip().map({
        "standard": 0,
        "express": 1,
        "priority": 2
    }))

    df['is_weekend_dispatch'] = (df['is_weekend_dispatch'].astype(str).str.lower().str.strip().map({
        "no": 0,
        "yes": 1
    }))

    df = df.drop(columns=unwanted_cols)
    df.to_csv(output_path, index=False)
    return df

def train_model(processed_path, model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns=['is_delayed'])
    y = df['is_delayed']

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def predict_new_data(model_path, input_path, processed_path, output_path):
    raw_df = pd.read_csv(input_path)
    raw_unique = raw_df.drop_duplicates().reset_index(drop=True)

    order_ids = raw_unique["order_id"].copy()

    cleaned_df = clean_data(
        input_path,
        processed_path,
    )

    model = joblib.load(model_path)
    predictions = model.predict(cleaned_df)

    result = pd.DataFrame({
        "order_id": order_ids.to_numpy(),
        "predicted_is_delayed": predictions,
    })

    result.to_csv(output_path, index=False)
    return result
