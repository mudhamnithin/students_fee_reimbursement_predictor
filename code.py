import streamlit as st
import pandas as pd
import joblib

# Load saved model
data = joblib.load("reimbursement_model.pkl")
model = data["model"]
mappings = data["mappings"]
feature_order = data["feature_order"]

st.title("🎓 Student Fee Reimbursement Prediction")
st.write("Fill the details to predict the reimbursement %")

inputs = {}

# Numerical inputs with unique keys
num_fields = {
    "Annual_Income": ("Annual Income (₹)", 0, 2000000, 500000),
    "Earning_Members": ("Earning Members", 1, 10, 1),
    "Rent_Amount": ("Rent Amount (₹)", 0, 200000, 0),
    "Loan_Burden": ("Loan Burden (₹)", 0, 500000, 0),
    "Medical_Expenditure": ("Medical Expenditure (₹)", 0, 300000, 0),
    "Children_Studying": ("Children Studying", 0, 5, 1),
    "Annual_College_Fee": ("Annual College Fee (₹)", 0, 300000, 0),
    "Transport_Cost": ("Transport Cost (₹)", 0, 100000, 0),
    "Land_Owned": ("Land Owned (Acres)", 0.0, 10.0, 0.0),
    "Agriculture_Income": ("Agriculture Income (₹)", 0, 300000, 0),
}

for field, (label, mn, mx, default) in num_fields.items():
    inputs[field] = st.number_input(
        label, mn, mx, default, key=f"num_input_{field}"
    )

# Categorical fields with unique keys
cat_fields = list(mappings.keys())

for col in cat_fields:
    inputs[col] = st.selectbox(
        col,
        list(mappings[col].keys()),
        key=f"cat_input_{col}"
    )

if st.button("Predict", key="predict_btn"):
    input_df = pd.DataFrame([inputs])

    # Apply label encoding
    for col, mapping in mappings.items():
        input_df[col] = input_df[col].map(mapping)

    # Ensure proper model column order
    input_df = input_df[feature_order]

    prediction = round(model.predict(input_df)[0], 2)

    st.success(f"✅ Predicted Reimbursement: **{prediction}%**")

    if prediction < 50:
        st.warning("Low reimbursement")
    elif prediction < 80:
        st.info("Medium reimbursement")
    else:
        st.success("High reimbursement ✅")
