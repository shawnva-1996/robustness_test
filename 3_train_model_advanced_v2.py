import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FILENAME = '3_processed_with_splits_full_engineered.csv'
TARGET_VARIABLE = 'log_likes'
# ---------------------

print("Starting Random Forest refinement process...")

# Load the dataset
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Successfully loaded '{INPUT_FILENAME}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    exit()

# Define features to drop
features_to_drop = [
    # --- Target and Direct Leaks ---
    'log_likes',        # The target variable itself
    'Like Count',       # The direct source of the target
    'Play Count',       # A post-event metric
    'viral',            # CRITICAL: This is a PROXY for the target

    # --- Snapshot-in-Time Leaks (Creator Stats) ---
    'Creator Follower Count',
    'Creator Following Count',
    'Creator Total Heart Count',
    'Creator Total Video Count',

    # --- Identifiers and High Cardinality / Raw Text ---
    'Video ID', 'Post Caption', 'Post Timestamp', 'Canonical URL',
    'Creator Username', 'Creator Nickname', 'Creator User ID', 'Creator SEC UID',
    'Music ID', 'Music Title', 'Music Author', 'Audio Play URL', 'Keywords',

    # --- Internal Housekeeping Columns ---
    '_parsed_time', 'cv_fold_grouped', 'temporal_split', 'is_holdout',
    'duration_s', # This is redundant with 'Video Duration (s)'
]

# Identify feature types automatically
X_temp = df.drop(columns=features_to_drop, errors='ignore')
categorical_features = X_temp.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_temp.select_dtypes(include=np.number).columns.tolist()

print(f"\nUsing {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")

# Split data based on the 'is_holdout' column
train_df = df[df['is_holdout'] == False].copy()
test_df = df[df['is_holdout'] == True].copy()

X_train = train_df[X_temp.columns]
y_train = train_df[TARGET_VARIABLE]
X_test = test_df[X_temp.columns]
y_test = test_df[TARGET_VARIABLE]

print(f"Training set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create the full pipeline with the Random Forest Regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])


# --- Step 1: Advanced Hyperparameter Tuning ---
print("\n--- Step 1: Running Advanced Hyperparameter Tuning ---")

# Define a wider range of parameters to test
param_dist = {
    'regressor__n_estimators': [100, 200, 300, 500],
    'regressor__max_depth': [10, 20, 30, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2', 1.0]
}

# Use RandomizedSearchCV to efficiently find the best parameters
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search.fit(X_train, y_train)

# This is our best, fine-tuned model
best_model = random_search.best_estimator_
best_params = random_search.best_params_

print(f"\nBest parameters found: {best_params}")
r2_score_full = best_model.score(X_test, y_test)
print(f"R2 score of the best model on test data: {r2_score_full:.4f}")


# --- Step 2: Feature Selection Based on Importance ---
print("\n--- Step 2: Running Feature Selection ---")

# 1. Get feature names and their importance scores from the best model
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances = best_model.named_steps['regressor'].feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# 2. Select the top N features (e.g., top 20)
N = 20
top_features = importance_df.head(N)['feature'].tolist()
print(f"Selected the top {N} most important features.")

# 3. Create new training and testing sets with only these top features
X_train_transformed = best_model.named_steps['preprocessor'].transform(X_train)
X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

# --- FIX: Use the standard DataFrame constructor which handles both sparse and dense arrays ---
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

X_train_simple = X_train_transformed_df[top_features]
X_test_simple = X_test_transformed_df[top_features]

# 4. Retrain a new model on the simpler data using the best parameters
regressor_best_params = {k.replace('regressor__', ''): v for k, v in best_params.items()}
simple_model = RandomForestRegressor(random_state=42, n_jobs=-1, **regressor_best_params)
simple_model.fit(X_train_simple, y_train)

# 5. Evaluate the simpler model and compare
simple_r2_test = simple_model.score(X_test_simple, y_test)
print(f"\nOriginal R2 score with all features: {r2_score_full:.4f}")
print(f"R2 score with top {N} features only: {simple_r2_test:.4f}")


# --- Step 3: Error Analysis ---
print("\n--- Step 3: Analyzing Top Prediction Errors ---")

# 1. Get predictions and calculate the error
predictions = best_model.predict(X_test)
test_df_with_errors = X_test.copy()
test_df_with_errors['actual_log_likes'] = y_test
test_df_with_errors['predicted_log_likes'] = predictions
test_df_with_errors['error'] = test_df_with_errors['actual_log_likes'] - test_df_with_errors['predicted_log_likes']

# 2. Sort to see the biggest mistakes
biggest_mistakes = test_df_with_errors.sort_values(by='error', key=abs, ascending=False)

print("Top 5 biggest prediction mistakes (where the model was most wrong):")
print(biggest_mistakes.head(5)[['actual_log_likes', 'predicted_log_likes', 'error']])