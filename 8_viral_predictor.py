import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FILENAME = '7b_data_with_full_transcript_features.csv'
TARGET_VARIABLE = 'viral' # Switched to classification
# ---------------------

print(f"Starting classification modeling on: '{INPUT_FILENAME}'")
print(f"Goal: Identify features that separate viral from non-viral videos.")

# Load the dataset
try:
    df = pd.read_csv(INPUT_FILENAME)
    # Drop rows where the target is missing, if any
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)
    print(f"Successfully loaded file. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    exit()

# --- CRITICAL: Define all features to drop ---
# This is the new, correct "master list"
features_to_drop = [
    # --- Target and Direct Leaks ---
    'log_likes',        # The regression target
    'Like Count',       # The direct source of the target
    'Play Count',       # A post-event metric
    'log1p_play_count', # A transformation of a leaky feature

    # --- Snapshot-in-Time Leaks (Creator Stats) ---
    'Creator Follower Count',
    'Creator Following Count',
    'Creator Total Heart Count',
    'Creator Total Video Count',

    # --- Identifiers and High Cardinality / Raw Text ---
    'Video ID', 'Post Caption', 'Post Timestamp', 'Canonical URL',
    'Creator Username', 'Creator Nickname', 'Creator User ID', 'Creator SEC UID',
    'Music ID', 'Music Title', 'Music Author', 'Audio Play URL', 'Keywords',
    'transcript',       # Drop the raw transcript text

    # --- Internal Housekeeping Columns ---
    '_parsed_time', 'cv_fold_grouped', 'temporal_split', 'is_holdout',
    'duration_s', # This is redundant
    
    # --- This script's target ---
    TARGET_VARIABLE # This ensures 'viral' (the target) is not in the features
]

# Identify feature types automatically
X_temp = df.drop(columns=features_to_drop, errors='ignore')
categorical_features = X_temp.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X_temp.select_dtypes(include=np.number).columns.tolist()
print(f"\nUsing {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")

# Split data
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

# --- Define Classification Models and Parameter Grids ---
param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__solver': ['liblinear']
    },
    'RandomForestClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None]
    },
    'LGBMClassifier': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [31, 50]
    }
}

models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=-1),
    'LGBMClassifier': lgb.LGBMClassifier(random_state=42, n_jobs=-1)
}

best_model = None
best_model_name = ""
best_score = -1

# Loop through models
for name, model in models.items():
    print(f"\n--- Tuning and Training {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grids[name],
        n_iter=10,
        cv=3,
        scoring='accuracy', # Evaluate based on accuracy
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    
    score = random_search.score(X_test, y_test)
    print(f"Best parameters for {name}: {random_search.best_params_}")
    print(f"Test Accuracy for {name}: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_model = random_search.best_estimator_
        best_model_name = name

print(f"\n--- Best Performing Model: {best_model_name} with Accuracy: {best_score:.4f} ---")

# --- Analyze the Best Model ---
y_pred = best_model.predict(X_test)
class_labels = np.unique(np.concatenate((y_train, y_test))) # Get all possible labels

# 1. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=class_labels, zero_division=0))

# 2. Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

# 3. Feature Importance (for tree-based models)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    print("\n--- Key Features That Determine Virality ---")
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save feature importances to CSV
    importance_df.to_csv('feature_importances.csv', index=False)
    print("Full feature importance list saved to 'feature_importances.csv'")
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))