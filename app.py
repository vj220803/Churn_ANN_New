import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load trained model and scaler
model = tf.keras.models.load_model("churn_model.keras")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Predict if a customer is likely to churn based on their details.")

# ---------------------------
# User Input
# ---------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=10000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# ---------------------------
# Encode categorical variables
# ---------------------------
# Gender: Male=1, Female=0
gender_map = {"Male": 1, "Female": 0}

# Geography: One-hot encoding (same as training)
geography_map = {
    "France": [1, 0, 0],
    "Germany": [0, 1, 0],
    "Spain": [0, 0, 1]
}

# ---------------------------
# Prepare numeric input (exact order of training)
# ---------------------------
numeric_values = [
    credit_score,
    age,
    tenure,
    balance,
    num_products,
    has_cr_card,
    is_active_member,
    estimated_salary
]

# Scale ONLY numeric values
numeric_scaled = scaler.transform([numeric_values])[0]

# Combine final input: scaled numeric + gender + geography
final_input = np.concatenate([
    numeric_scaled,
    [gender_map[gender]],
    geography_map[geography]
])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn"):
    prediction = model.predict(final_input.reshape(1, -1))[0][0]
    if prediction > 0.5:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {prediction:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn. (Probability: {prediction:.2f})")
