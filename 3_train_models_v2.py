import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import numpy as np
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FILENAME = '3_processed_with_splits_full_engineered.csv'
OUTPUT_FILENAME = '3_modelling_results_engineered.csv'
TARGET_VARIABLE = 'log_likes'
# ---------------------

print("Starting model training on engineered features...")

# Load the dataset
try:
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Successfully loaded '{INPUT_FILENAME}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    exit()

# --- Define features to drop ---
# We remove identifiers, text, dates, the target itself, and potential data leakage features.
# 'log1p_play_count' is a direct transformation of Play Count and is removed to prevent leakage.
# features_to_drop = [
#     'Video ID', 'Post Caption', 'Post Timestamp', 'Canonical URL', 'Play Count', 'Like Count',
#     'Creator Username', 'Creator Nickname', 'Creator User ID', 'Creator SEC UID',
#     'Music ID', 'Music Title', 'Music Author', 'Audio Play URL', 'Keywords',
#     '_parsed_time', 'cv_fold_grouped', 'temporal_split', 'is_holdout',
#     'log1p_play_count', # Potential data leakage
#     TARGET_VARIABLE
# ]

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

# Identify categorical and numerical features automatically from the remaining columns
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

# --- Define parameter grids for the selected models ---
param_grids = {
    'Ridge': {'regressor__alpha': [1.0, 10.0, 50.0]},
    'Lasso': {'regressor__alpha': [0.001, 0.01, 0.1]},
    'ElasticNet': {'regressor__alpha': [0.1, 1.0], 'regressor__l1_ratio': [0.5, 0.9]},
    'RandomForest': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20]
    },
    'SVR': {
        'regressor__C': [0.1, 1.0, 10.0],
        'regressor__kernel': ['linear', 'rbf']
    },
    'LightGBM': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__num_leaves': [31, 50]
    }
}

# Define the models
models = {
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'SVR': SVR(),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
}

# Store results
results = []

# Loop through each model to tune, train, and evaluate
for name, model in models.items():
    print(f"\n--- Tuning and Training {name} ---")
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    print(f"R2 Train: {r2_train:.4f}")
    print(f"R2 Test: {r2_test:.4f}")
    
    result_row = {
        'model': name,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'best_params': str(grid_search.best_params_),
        'top_10_features': None
    }
    
    if name in ['LightGBM', 'RandomForest']:
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        importances = best_model.named_steps['regressor'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_10 = importance_df.head(10)
        top_features_str = ', '.join([f"{row.feature} ({row.importance:.3f})" for index, row in top_10.iterrows()])
        
        result_row['top_10_features'] = top_features_str

    results.append(result_row)

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Modeling complete. All results saved to '{OUTPUT_FILENAME}'.")
print("\n--- Final Consolidated Results ---")
print(results_df)