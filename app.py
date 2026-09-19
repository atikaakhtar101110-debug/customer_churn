import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Page configuration MUST be the very first Streamlit command
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Prediction System")
st.markdown("Enter customer details below to predict whether they are likely to churn.")

@st.cache_resource
def load_or_train_model():
    model_filename = "customer_churn_xgb_model.pkl"
    
    # 1. Try loading the pre-trained model file if it exists
    if os.path.exists(model_filename):
        try:
            loaded_model = joblib.load(model_filename)
            return loaded_model
        except Exception as e:
            st.warning(f"⚠️ Could not load saved model ({e}). Training a fresh fallback model...")

    # 2. Fallback: Train a model instantly on startup so the UI never crashes
    df_dummy = pd.DataFrame({
        'Age': [25, 45, 35, 52, 23, 40],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
        'Tenure': [2, 24, 12, 60, 1, 18],
        'MonthlyCharges': [29.9, 89.5, 70.0, 105.0, 45.0, 75.0],
        'TotalCharges': [59.8, 2148.0, 840.0, 6300.0, 45.0, 1350.0],
        'Contract': ['Month-to-month', 'Two month', 'One month', 'Two month', 'Month-to-month', 'One month'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check', 'Credit card (automatic)'],
        'Churn': ['Yes', 'No', 'No', 'No', 'Yes', 'No']
    })
    
    X = df_dummy.drop(columns=['Churn'])
    y = df_dummy['Churn'].map({'Yes': 1, 'No': 0})
    
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
    
    # Fit the pipeline on the fallback data
    pipeline.fit(X, y)
    return pipeline

# Load or initialize the pipeline safely
pipeline = load_or_tech_model = load_or_train_model()

# User Input Form
with st.form("churn_form"):
    st.subheader("Customer Demographics & Account Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
        
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)
        contract = st.selectbox("Contract Type", ["One month", "Two month", "Month-to-month"])
        
    payment_method = st.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    
    submit_button = st.form_submit_button(label="Predict Churn")

# Prediction logic
if submit_button:
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'TotalCharges': [total_charges]
    })
    
    try:
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        
        st.divider()
        st.subheader("Prediction Results")
        
        if prediction == 1 or probability > 0.5:
            st.error(f"⚠️ **High Risk of Churn!** (Probability: {probability:.2%})")
            st.markdown("This customer shows strong indicators of leaving. Consider offering a retention discount or contract upgrade.")
        else:
            st.success(f"✅ **Low Risk of Churn** (Probability: {probability:.2%})")
            st.markdown("This customer is likely to stay loyal.")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
