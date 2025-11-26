import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FILENAME = '8_data_with_performance_category.csv'
OUTPUT_FILENAME = '9_final_classification_results.csv'
TARGET_VARIABLE = 'performance_category' # Switched to the new binary target
# ---------------------

print(f"Starting final classification modeling on: '{INPUT_FILENAME}'")
print(f"Goal: Predict '{TARGET_VARIABLE}' and identify key drivers.")

# Load the dataset
try:
    df = pd.read_csv(INPUT_FILENAME)
    df.dropna(subset=[TARGET_VARIABLE], inplace=True)
    print(f"Successfully loaded file. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    exit()

# --- CRITICAL: Define all features to drop to prevent any data leakage ---
# features_to_drop = [
#     'performance_category', # The target variable itself
#     'viral',                # The original source of the target
#     'log_likes',            # A direct LEAK for the performance category
#     'Like Count',           # Direct leak
#     'Play Count',           # Direct leak
#     'log1p_play_count',     # Direct leak

#     'Video ID', 'Post Caption', 'Post Timestamp', 'Canonical URL',
#     'Creator Username', 'Creator Nickname', 'Creator User ID', 'Creator SEC UID',
#     'Music ID', 'Music Title', 'Music Author', 'Audio Play URL', 'Keywords',
#     'transcript',
#     '_parsed_time', 'cv_fold_grouped', 'temporal_split', 'is_holdout',
#     'duration_s',
# ]

features_to_drop = [
    # --- Target and Direct Leaks ---
    'log_likes',        # The regression target
    'Like Count',       # The direct source of the target
    'Play Count',       # A post-event metric
    'log1p_play_count', # A transformation of a leaky feature
    'viral',            # <-- THE LEAK! The source of your target.

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
    TARGET_VARIABLE # This ensures 'performance_category' is not in the features
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

# --- Define CLASSIFICATION Models and Parameter Grids ---
# Note: We are now using classifiers like LogisticRegression and RandomForestClassifier
param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.1, 1.0, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    },
    'RandomForestClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, None],
        'classifier__class_weight': ['balanced', None]
    },
    'SVC': {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf']
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
    'SVC': SVC(random_state=42),
    'LGBMClassifier': lgb.LGBMClassifier(random_state=42, n_jobs=-1)
}


best_model = None
best_model_name = ""
best_score = -1

# Loop through models
for name, model in models.items():
    print(f"\n--- Tuning and Training {name} ---")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    # Using F1-score for evaluation is often better for imbalanced datasets
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grids[name],
        n_iter=10,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    
    # Use the main score for comparison
    test_score = random_search.score(X_test, y_test)
    
    print(f"Best parameters for {name}: {random_search.best_params_}")
    print(f"Test F1 Score for {name}: {test_score:.4f}")
    
    if test_score > best_score:
        best_score = test_score
        best_model = random_search.best_estimator_
        best_model_name = name

print(f"\n--- Best Performing Model: {best_model_name} with F1 Score: {best_score:.4f} ---")

# --- Analyze the Best Model ---
y_pred = best_model.predict(X_test)
class_labels = best_model.classes_

# 1. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 2. Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.savefig('final_confusion_matrix.png')
print("Confusion matrix plot saved as 'final_confusion_matrix.png'")

# 3. Feature Importance (for tree-based models)
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    print("\n--- Key Features That Distinguish High vs. Low Performance ---")
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['classifier'].feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('final_feature_importances.csv', index=False)
    print("Full feature importance list saved to 'final_feature_importances.csv'")
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))