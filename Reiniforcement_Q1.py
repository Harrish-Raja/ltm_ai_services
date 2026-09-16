import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
 
def preprocess_data(dataframe):
    df = dataframe.copy()
    df = dataframe.drop(columns='PolicyID')
    num = df.select_dtypes(include=['number']).columns.tolist()
    cat = df.select_dtypes(include=['object']).columns.tolist()
    for col in num:
        df[col] = df[col].fillna(df[col].median())
    for co in cat:
        if not df[co].mode().empty:
            df[co] = df[co].fillna(df[co].mode()[0])
    if cat:
        df = pd.get_dummies(df,columns=cat,drop_first=True)
   
    return df
 
def train_model(cleaned_data):
    df = cleaned_data.copy()
    features = df.drop(columns='ClaimCost')
    target = df['ClaimCost']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    dump(model, "random_forest_model.joblib")
    return model, features.columns
 
def predict_claims(model, prediction_data, training_columns):
    df = preprocess_data(prediction_data)
    df = df.reindex(columns=training_columns,fill_value=0)
    pre = model.predict(df)
    result_df = prediction_data.copy()
    result_df['PredictedClaimCost'] = pre.round(2)
    result_df.to_csv("predicted_claims.csv",index=False)
    return result_df
