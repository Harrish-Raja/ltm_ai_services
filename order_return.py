import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 

def clean_data(input_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=['order_id','order_date','center_code','destination_city','carrier_partner'])
    
    n_cols = ['package_weight_kg','item_price']
    for i in n_cols:
        df[i] = df[i].fillna(df[i].median())
        
    df['shipping_method'] = df['shipping_method'].str.lower().str.strip().map({
        "standard":0,
        "express":1,
        "priority":2
    })
    
    df['is_gift_order'] = df['is_gift_order'].str.lower().str.strip().map({
        "no":0,
        "yes":1
    })
    
    df.to_csv(output_path,index=False)
    return df 
    
    
def train_model(processed_path,model_path):
    df = pd.read_csv(processed_path)
    X = df.drop(columns=['is_returned'])
    y = df['is_returned']
    model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
    model.fit(X,y)
    joblib.dump(model,model_path)
    return model 
    
    
def predict_new_data(model_path,input_path,processed_path,output_path):
    df = pd.read_csv(input_path)
    df=df.drop_duplicates().reset_index(drop=True)
    ids = df['order_id'].copy()
    
    cleaned_df = clean_data(input_path,processed_path)
    
    model = joblib.load(model_path)
    predictions = model.predict(cleaned_df)
    
    result = pd.DataFrame({
        "order_id":ids.to_numpy(),
        "predicted_is_returned":predictions
    })
    
    result.to_csv(output_path,index=False)
    return result