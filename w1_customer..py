

#customer.py
    import pandas as pd 
    import joblib 
    from sklearn.ensemble import RandomForestClassifier 

    def clean_data(input_path,output_path):
        df = pd.read_csv(input_path)
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.drop(columns=['customer_id','signup_date','account_manager','billing_region','subscription_channel'])

        n_cols = ['monthly_charges','tenure_months']
        for i in n_cols:
            df[i] = df[i].fillna(df[i].median())

        df['contract_type'] = df['contract_type'].str.lower().str.strip().map({
            "month-to-month":0,
            "one-year":1,
            "two-year":2
        })

        df['auto_pay'] = df['autopay'].str.lower().str.strip().map({
            "no":0,
            "yes":1
        })

        df.to_csv(output_path,index=False)
        return df 

    def train_model(processed_path,model_path):
        df = pd.read_csv(processed_path)
        X = df.drop(columns=['churned'])
        y = df['churned']
        model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
        model.fit(X,y)
        joblib.dump(model,model_path)
        return model 

    def predict_new_data(model_path,input_path,processed_path,output_path):
        df = pd.read_csv(input_path)
        df = df.drop_duplicates().reset_index(drop=True)
        id = df['customer_id'].copy()

        cleaned_df = clean_data(input_path,processed_path)

        model = joblib.load(model_path)
        predictions = model.predict(cleaned_df)

        result = pd.DataFrame({
            "customer_id":id.to_numpy(),
            "predicted_churned":predictions
        })
        result.to_csv(output_path,index=False)
        return result 
