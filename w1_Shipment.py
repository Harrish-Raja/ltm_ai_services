#shipment.py 
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 

def clean_data(input_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=['order_id','dispatch_date','origin_warehouse_code','destination_city','courier_partner'])
    n_col = ['distance_km','package_weight_kg']
    for i in n_col:
        df[i] = df[i].fillna(df[i].median())
    df['shipping_speed'] = df['shipping_speed'].str.lower().str.strip().map(
        {
            "standard":0,
            "express":1,
            "priority":2
        }
    )

    df['is_weekend_dispatch'] = df['is_weekend_dispatch'].str.lower().str.strip().map({
        "no":0,
        'yes':1
    })

    df.to_csv(output_path,index=False)
    return df 


def train_model(processed_path,model_path):
    df = pd.read_csv(processed_path)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=['is_delayed'])
    y = df['is_delayed']
    model = RandomForestClassifier(
        n_estimators = 100 , max_depth = 10, random_state=42
    )
    model.fit(X,y)
    joblib.dump(model,model_path)
    return model

def predict_new_data(model_path,input_path,processed_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)
    order_ids = df['order_id'].copy()

    cleaned_df = clean_data(input_path,processed_path)
    model = joblib.load(model_path)
    predictions = model.predict(cleaned_df)

    result = pd.DataFrame({
        "order_id":order_ids.to_numpy(),
        "predicted_is_delayed":predictions
    })

    result.to_csv(output_path,index=False)
    return result
