# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
data = joblib.load("reimbursement_model.pkl")
model = data["model"]
mappings = data["mappings"]
feature_order = data["feature_order"]

st.title("🎓 Student Fee Reimbursement Prediction")
st.write("Fill the details below to predict eligibility for reimbursement.")

# Input form
inputs = {}
inputs["Annual_Income"] = st.number_input("Annual Income", 0, 10000000, 500000)
inputs["Earning_Members"] = st.number_input("Earning Members", 0, 10, 1)
inputs["Rent_Amount"] = st.number_input("Rent Amount", 0, 200000, 50000)
inputs["Loan_Burden"] = st.number_input("Loan Burden", 0, 200000, 0)
inputs["Medical_Expenditure"] = st.number_input("Medical Expenditure", 0, 200000, 0)
inputs["Children_Studying"] = st.number_input("Children Studying", 0, 10, 2)
inputs["Annual_College_Fee"] = st.number_input("Annual College Fee", 0, 500000, 100000)
inputs["Transport_Cost"] = st.number_input("Transport Cost", 0, 100000, 0)
inputs["Land_Owned"] = st.number_input("Land Owned (in acres)", 0.0, 100.0, 0.0)
inputs["Agriculture_Income"] = st.number_input("Agriculture Income", 0, 500000, 0)

# Dropdowns
for col, mapping in mappings.items():
    inputs[col] = st.selectbox(col, list(mapping.keys()))

if st.button("Predict Reimbursement"):
    input_df = pd.DataFrame([inputs])

    # Apply mappings
    for col, mapping in mappings.items():
        input_df[col] = input_df[col].map(mapping)

    # 🔹 Ensure same feature order as training
    input_df = input_df[feature_order]

    # Predict
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    result = classes[probs.argmax()]

    # Bonus rule: single parent / orphan adjustment
    parent_type = input_df["Parent_Type"].iloc[0]
    if parent_type in [1, 2] and result == 50:
        result = 100
    elif parent_type == 3:
        result = 100

    # Display result
    st.success(f"✅ Predicted Reimbursement: {result}%")
    st.write("📊 Class Probabilities:")
    st.json({f"{cls}%": f"{prob*100:.2f}%" for cls, prob in zip(classes, probs)})
