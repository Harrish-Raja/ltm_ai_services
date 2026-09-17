
#employee_attrition

import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier 

def clean_data(input_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=['employee_id','hire_date','manager_name','department_code','work_location'])

    n_col = ['monthly_salary','years_at_company']
    for i in n_col:
        df[i] = df[i].fillna(df[i].median())

    df['job_level'] = df['job_level'].str.lower().str.strip().map({
        "junior":0,
        "mid":1,
        "senior":2
    })

    df['remote_work'] = df['remote_work'].str.lower().str.strip().map({
        "no":0,
        "yes":1
    })


    df.to_csv(output_path,index=False)

    return df


def train_model(processed_path,model_path):
    df = pd.read_csv(processed_path)

    X = df.drop(columns=['attrited'])
    y = df['attrited']

    model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
    model.fit(X,y)
    joblib.dump(model,model_path)
    return model

def predict_new_data(model_path,input_path,processed_path,output_path):
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().reset_index(drop=True)

    id = df['employee_id'].copy()

    cleaned_df = clean_data(input_path,processed_path)

    model = joblib.load(model_path)
    predictions = model.predict(cleaned_df)

    result = pd.DataFrame({
        "employee_id":id.to_numpy(),
        "predicted_attrited":predictions
    })

    result.to_csv(output_path,index=False)
    return result
