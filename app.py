"""
Title: Customer Churn Prediction System (FastAPI + Streamlit in One)
Filepath: app.py

Description:
This self-contained application provides:
1. A FastAPI backend service with endpoints to predict customer churn using an XGBoost pipeline.
2. A Streamlit interactive frontend where users can input customer data (Tenure, Charges, Contract, etc.)
   and receive real-time predictions with probability scores and visual indicators.

Instructions to run:
1. Save this script as `app.py`.
2. Install dependencies: `pip install fastapi uvicorn streamlit requests scikit-learn xgboost pandas`
3. Run the application launcher block at the bottom, or run them in separate terminals:
   - Backend: `uvicorn app:app --reload --port 8000`
   - Frontend: `streamlit run app.py` (when running Streamlit standalone, it detects the UI mode automatically).
"""

import os
import sys
import threading
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import streamlit as st
import requests
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using XGBoost and scikit-learn pipelines.",
    version="1.0.0"
)

# Enable CORS for local client connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# In a production environment, you would load a pickled model using joblib.
# Here, we dynamically create and train a dummy pipeline on startup using synthetic data 
# that mirrors your column structure so the app runs out-of-the-box.

def create_trained_pipeline():
    # Synthetic mock training dataframe matching user's columns
    np.random.seed(42)
    n_samples = 300
    df_mock = pd.DataFrame({
        'Age': np.random.randint(18, 70, size=n_samples),
        'Tenure': np.random.randint(1, 72, size=n_samples),
        'MonthlyCharges': np.random.uniform(20.0, 120.0, size=n_samples),
        'TotalCharges': np.random.uniform(100.0, 8000.0, size=n_samples),
        'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples),
        'Churn': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
    })

    X = df_mock.drop(columns=['Churn'])
    y = df_mock['Churn'].map({'Yes': 1, 'No': 0})

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

    pipeline.fit(X, y)
    return pipeline

# Global model instance
model_pipeline = create_trained_pipeline()

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
class CustomerData(BaseModel):
    Age: int
    Tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Gender: str
    Contract: str
    PaymentMethod: str

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API is running. Go to /docs for Swagger UI."}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert incoming request payload into a pandas DataFrame matching feature layout
    input_data = pd.DataFrame([{
        'Age': customer.Age,
        'Tenure': customer.Tenure,
        'MonthlyCharges': customer.MonthlyCharges,
        'TotalCharges': customer.TotalCharges,
        'Gender': customer.Gender,
        'Contract': customer.Contract,
        'PaymentMethod': customer.PaymentMethod
    }])

    # Predict probability and class label
    prediction = int(model_pipeline.predict(input_data)[0])
    probabilities = model_pipeline.predict_proba(input_data)[0]
    churn_probability = float(probabilities[1])

    result_label = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": result_label,
        "churn_probability": round(churn_probability * 100, 2),
        "retention_probability": round(probabilities[0] * 100, 2)
    }

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def run_streamlit():
    st.set_page_config(
        page_title="Customer Churn Intelligence Portal",
        page_icon="🔄",
        layout="wide"
    )

    # Custom styling injection
    st.markdown("""
        <style>
        .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; }
        .sub-text { color: #4B5563; font-size: 1.1rem; }
        .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #2563EB; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">Customer Churn Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Enter customer profile details below to evaluate the likelihood of churn using our XGBoost backend pipeline.</p>', unsafe_allow_html=True)
    
    st.divider()

    # Create input form layout using columns
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics & Account")
            age = st.slider("Customer Age", 18, 90, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        with col2:
            st.subheader("Billing Information")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=250.0, value=70.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=850.0)

        with col3:
            st.subheader("Contract & Payment")
            contract = st.selectbox("Contract Duration", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])

        submitted = st.form_submit_button("Analyze Churn Risk", use_container_width=True)

    if submitted:
        payload = {
            "Age": age,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Gender": gender,
            "Contract": contract,
            "PaymentMethod": payment_method
        }

        # API Request endpoint (default FastAPI local server port)
        api_url = "http://localhost:8000/predict"

        with st.spinner("Connecting to FastAPI backend & evaluating risk score..."):
            try:
                response = requests.post(api_url, json=payload, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    pred = data["churn_prediction"]
                    prob = data["churn_probability"]

                    st.divider()
                    st.subheader("Prediction Result")

                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        if pred == "Yes":
                            st.error(f"### High Risk: Customer Likely to Churn")
                        else:
                            st.success(f"### Low Risk: Customer Likely to Stay")

                    with res_col2:
                        st.metric(label="Calculated Churn Probability", value=f"{prob}%")
                        st.progress(int(prob))

                else:
                    st.error(f"Backend error code: {response.status_code}. Make sure FastAPI is running!")
            except requests.exceptions.ConnectionError:
                st.error("Connection Refused! Please ensure the FastAPI backend server is running on port 8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if script is executed via Streamlit CLI or Python direct execution
    if "streamlit" in sys.argv[0] or os.environ.get("STREAMLIT_RUN") == "true":
        run_streamlit()
    else:
        print("Starting FastAPI background server and launching Streamlit user interface...")
        # Start FastAPI server in a background thread for convenience
        import uvicorn
        
        def run_fastapi():
            uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

        api_thread = threading.Thread(target=run_fastapi, daemon=True)
        api_thread.start()

        # Execute streamlit UI via command process
        os.environ["STREAMLIT_RUN"] = "true"
        os.system(f"streamlit run {__file__}")
