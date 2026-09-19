@st.cache_resource
def load_or_train_model():
    model_filename = "customer_churn_xgb_model.pkl"
    
    # Try loading the saved model first, with a safety catch for version mismatches
    if os.path.exists(model_filename):
        try:
            return joblib.load(model_filename)
        except Exception as e:
            st.warning(f"⚠️ Could not load saved model due to version mismatch ({e}). Initializing pipeline...")

    # Fallback: Define preprocessor and pipeline structure if no model file exists
    numeric_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['Gender', 'Contract', 'PaymentMethod']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    return pipeline
