# Notebooks/1_model_training.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sklearn
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, recall_score
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("--- STARTING MODEL TRAINING PIPELINE ---")

# === 1. Load Datasets ===
print("STEP 1: Loading datasets...")
try:
    # IMPORTANT: Assumes the training script is run from the 'notebooks' directory
    meta_df = pd.read_excel(r"C:\Users\Dell\Movie project\DATA\Final Dataset.xlsx")
    bert_df = pd.read_excel(r"C:\Users\Dell\Movie project\DATA\longformer_embeddings.xlsx")
except FileNotFoundError:
    print("Error: Make sure 'Final Dataset.xlsx' and 'longformer_embeddings.xlsx' are in the project's root directory.")
    exit()

bert_df = bert_df.drop(columns=["Plot Synopsis", "Title"], errors="ignore")

# === 2. Preprocessing and Feature Engineering ===
print("STEP 2: Preprocessing and feature engineering...")
categorical_vars = ["MPA", "1st Genre"]
meta_df["MPA"] = meta_df["MPA"].apply(lambda x: x if x in ["PG-13", "R"] else "Other")
main_genres = ["Action", "Drama", "Comedy", "Biography"]
meta_df["1st Genre"] = meta_df["1st Genre"].apply(lambda x: x if x in main_genres else "Other")

# OneHotEncoder for categorical variables
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(meta_df[categorical_vars])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_vars))

# Numerical variables - Assumes 'Director Avg' is in your dataset
numerical_vars = ["budget", "Duration_Minutes", "First Actor Avg", "Second Actor Avg", "Average IMDb Rating"]
numerical_df = meta_df[numerical_vars].reset_index(drop=True)

# SVD for BERT embeddings
bert_scaler = StandardScaler()
bert_scaled = bert_scaler.fit_transform(bert_df.values)
svd = TruncatedSVD(n_components=150, random_state=42)
bert_svd = svd.fit_transform(bert_scaled)
bert_svd_df = pd.DataFrame(bert_svd, columns=[f"bert_svd_{i}" for i in range(bert_svd.shape[1])])

# Combine all features
X = pd.concat([numerical_df, encoded_cat_df.reset_index(drop=True), bert_svd_df.reset_index(drop=True)], axis=1)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(meta_df["Rating"].apply(lambda r: "Success" if r >= 6.5 else "Unsuccess"))

# Get feature names in the correct order for the API
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === 3. Tune LGBM for Recall ===
print("STEP 3: Tuning LGBM model for Class 0 Recall...")
recall_scorer_0 = make_scorer(recall_score, pos_label=0)
param_grid = {'class_weight': [{0: w, 1: 1} for w in np.arange(1.0, 2.5, 0.2)]}
grid_search = GridSearchCV(
    LGBMClassifier(random_state=42, n_jobs=-1),
    param_grid, scoring=recall_scorer_0, cv=3, n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_lgbm_params = grid_search.best_params_
print(f"Best class_weight found: {best_lgbm_params['class_weight']}")

# === 4. Build and Train Final Stacked Model ===
print("STEP 4: Building and training final stacked model...")
xgb_params = {
    'colsample_bytree': 0.6727, 'learning_rate': 0.0467, 'max_depth': 5,
    'n_estimators': 100, 'reg_alpha': 0.5247, 'reg_lambda': 0.8638,
    'subsample': 0.7165, 'random_state': 42, 'n_jobs': -1,
    'use_label_encoder': False, 'eval_metric': 'logloss'
}
base_xgb = XGBClassifier(**xgb_params)
base_lgbm_tuned = LGBMClassifier(random_state=42, n_jobs=-1, **best_lgbm_params)

calibrated_xgb = CalibratedClassifierCV(base_xgb, method='isotonic', cv=3)
calibrated_lgbm = CalibratedClassifierCV(base_lgbm_tuned, method='isotonic', cv=3)

estimators = [('xgb', calibrated_xgb), ('lgbm', calibrated_lgbm)]
final_model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(), cv=3
)
final_model.fit(X_train, y_train)

# === 5. Save Artifacts ===
print("STEP 5: Saving all model artifacts...")
artifacts_dir = "../artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

joblib.dump(final_model, os.path.join(artifacts_dir, "stacked_model.joblib"))
joblib.dump(encoder, os.path.join(artifacts_dir, "one_hot_encoder.joblib"))
joblib.dump(bert_scaler, os.path.join(artifacts_dir, "bert_scaler.joblib"))
joblib.dump(svd, os.path.join(artifacts_dir, "svd_transformer.joblib"))
joblib.dump(target_encoder, os.path.join(artifacts_dir, "target_encoder.joblib"))
joblib.dump(feature_names, os.path.join(artifacts_dir, "feature_names.joblib"))

print("\n --- MODEL TRAINING COMPLETE --- ")
print(f"All artifacts saved to '{artifacts_dir}' folder.")