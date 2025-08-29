# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("student_reimbursement_data.csv")

# Mapping
mappings = {
    "Parent_Type": {"Both Alive": 0, "Single Mother": 1, "Single Father": 2, "Orphan": 3},
    "House_Type": {"Own": 0, "Rent": 1},
    "Parent_Occupation": {"Unemployed": 0, "Daily Wage": 1, "Private Job": 2, "Govt Job": 3, "Business": 4, "Agriculture": 5, "Foreign": 6},
    "Job_Type": {"Temporary": 0, "Contract": 1, "Permanent": 2, "Self-employed": 3},
    "Any_Disability": {"No": 0, "Yes": 1},
    "Hostel_or_DayScholar": {"DayScholar": 0, "Hostel": 1},
    "Urban_Rural": {"Urban": 0, "Rural": 1},
    "First_Generation_Learner": {"no": 0, "yes": 1}
}

for col, mapping in mappings.items():
    df[col] = df[col].map(mapping)

# Features & Target
X = df.drop("Reimbursement_%", axis=1)
y = df["Reimbursement_%"]

# Oversample Single Parent / Orphan
boost_df = df[df["Parent_Type"].isin([1, 2, 3])].copy()
df_balanced = pd.concat([df, boost_df, boost_df])

X = df_balanced.drop("Reimbursement_%", axis=1)
y = df_balanced["Reimbursement_%"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model + mappings + feature order
joblib.dump({
    "model": model,
    "mappings": mappings,
    "feature_order": list(X.columns)   # 🔹 save feature names + order
}, "reimbursement_model.pkl")

print("✅ Model & feature order saved!")
