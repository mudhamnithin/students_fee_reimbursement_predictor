import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Load Data
df = pd.read_csv("student_reimbursement_data.csv")

target = "Reimbursement_%"
y = df[target]
X = df.drop(columns=[target])

# Categorical columns
cat_cols = ['Parent_Type','House_Type','Parent_Occupation','Any_Disability',
            'Urban_Rural','First_Generation_Learner','Job_Type','Hostel_or_DayScholar']

# Create categorical mappings
mappings = {}
for col in cat_cols:
    unique_vals = df[col].dropna().unique()
    mappings[col] = {v: i for i, v in enumerate(unique_vals)}
    df[col] = df[col].map(mappings[col])

X = df.drop(columns=[target])
y = df[target]

# ✅ Important fairness-weighted features
important_features = [
    "Parent_Type",
    "Medical_Expenditure",
    "Agriculture_Income",
    "Annual_Income",
    "Any_Disability"
]

# ✅ Create feature weights (3x for important features)
feature_weights = [3 if col in important_features else 1 for col in X.columns]

print("\n🎯 Feature Weights Applied:")
for col, w in zip(X.columns, feature_weights):
    print(f"{col}: {w}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=[X.columns.get_loc(c) for c in cat_cols])
test_pool = Pool(X_test, y_test, cat_features=[X.columns.get_loc(c) for c in cat_cols])

# ✅ Train CatBoost with fairness feature weights
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    feature_weights=feature_weights,
    verbose=False
)

model.fit(train_pool)

# Evaluate
preds = model.predict(test_pool)
print("\n📊 MODEL PERFORMANCE")
print("R2 Score:", r2_score(y_test, preds))
print("MAE:", mean_absolute_error(y_test, preds))

# Save dictionary
model_dict = {
    "model": model,
    "mappings": mappings,
    "feature_order": list(X.columns)
}

joblib.dump(model_dict, "reimbursement_model.pkl")
print("\n✅ Model saved as reimbursement_model.pkl with fairness weighting")
