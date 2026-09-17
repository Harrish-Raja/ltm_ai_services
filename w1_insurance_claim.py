#insurance_claim

import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 

def clean_data(input_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=['claim_id','filing_date','adjuster_code','claim_region','insurer_partner'])

    n_cols = ['claim_amount','vehicle_age']
    for i in n_cols:
        df[i] = df[i].fillna(df[i].median())

    df['claim_severity'] = df['claim_severity'].str.lower().str.strip().map({
        "minor":0,
        "moderate":1,
        "severe":2
    })

    df['prior_claims_flag'] = df['prior_claims_flag'].str.lower().str.strip().map({
        "no":0,
        "yes":1
    })


    df.to_csv(output_path,index=False)
    return df

def train_model(processed_path,model_path):
    df  =pd.read_csv(processed_path)

    X = df.drop(columns=['is_fraudulent'])
    y = df['is_fraudulent']

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X,y)
    joblib.dump(model,model_path)
    return model


def predict_new_data(model_path,input_path,processed_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)

    ids = df['claim_id'].copy()

    cleaned_df = clean_data(input_path,processed_path)

    model = joblib.load(model_path)
    predictions = model.predict(cleaned_df)

    result = pd.DataFrame({
        "claim_id":ids.to_numpy(),
        "predicted_is_fraudulent":predictions
    })

    result.to_csv(output_path,index=False)
    return result
