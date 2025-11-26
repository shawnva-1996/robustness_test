import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_FILENAME = '2_processed.csv'
OUTPUT_FILENAME = '2_processed_with_splits.csv'
TARGET_TIMESTAMP_COL = 'Post Timestamp'
TEST_SET_PERCENT = 0.2
N_CV_FOLDS = 5
# ---------------------

print(f"Loading data from '{INPUT_FILENAME}' to create temporal splits...")
try:
    df = pd.read_csv(INPUT_FILENAME)
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    print("Please run '2_drop_pre_post_features.py' first.")
    exit()

# --- 1. Ensure timestamp is a datetime object ---
if TARGET_TIMESTAMP_COL not in df.columns:
    print(f"Error: The critical column '{TARGET_TIMESTAMP_COL}' is not in the data.")
    print("This column is required for a temporal split.")
    exit()
    
df[TARGET_TIMESTAMP_COL] = pd.to_datetime(df[TARGET_TIMESTAMP_COL])

# --- 2. Sort dataframe by time ---
print("Sorting data by timestamp...")
df = df.sort_values(by=TARGET_TIMESTAMP_COL).reset_index(drop=True)

# --- 3. Create the main Train/Test (Holdout) split ---
print(f"Creating {TEST_SET_PERCENT*100}% holdout test set...")
split_index = int(len(df) * (1 - TEST_SET_PERCENT))
df['is_holdout'] = False
df.loc[split_index:, 'is_holdout'] = True

df['temporal_split'] = 'Train'
df.loc[df['is_holdout'] == True, 'temporal_split'] = 'Test'

# --- 4. Create Cross-Validation (CV) folds on the TRAINING data only ---
print(f"Creating {N_CV_FOLDS} sequential CV folds for the training set...")

# Initialize fold column
df['cv_fold_grouped'] = -1 # -1 indicates 'Test' or 'Unassigned'

# Get the indices for the training data
train_indices = df[df['is_holdout'] == False].index

# Use pd.qcut to create N sequential, roughly equal-sized time bins
# We use the *indices* as a proxy for time since we already sorted
fold_labels = pd.qcut(train_indices, N_CV_FOLDS, labels=range(1, N_CV_FOLDS + 1))

# Assign the fold labels back to the dataframe
df.loc[train_indices, 'cv_fold_grouped'] = fold_labels

# --- 5. Save the new dataframe ---
df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Successfully created temporal splits and saved to '{OUTPUT_FILENAME}'.")
print("\nSplit summary:")
print(df['temporal_split'].value_counts())
print("\nCV Fold summary (in training data):")
print(df[df['is_holdout'] == False]['cv_fold_grouped'].value_counts().sort_index())