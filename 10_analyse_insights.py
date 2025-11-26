import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
INPUT_FILE = '7b_data_with_full_transcript_features.csv'
TARGET_COLUMN = 'performance_category'
SOURCE_COLUMN = 'viral' # The original column we use to build the target
# ---------------------

print(f"--- Loading Insights from '{INPUT_FILE}' ---")

# 1. Load Data
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: File not found: {INPUT_FILE}")
    print("Please make sure you have run '7d_create_final_dataset.py' to create the final file.")
    exit()

# 2. Re-create the Target Variable (from script 9_high_vs_low_perf.py)
mapping = {
    'viral': 'High Performance',
    'mid-viral': 'High Performance',
    'non-viral': 'Low Performance'
}
df[TARGET_COLUMN] = df[SOURCE_COLUMN].map(mapping)

# Drop rows where mapping failed (e.g., 'viral' was NaN)
df.dropna(subset=[TARGET_COLUMN], inplace=True)

# 3. Separate DataFrames for comparison
high_perf_df = df[df[TARGET_COLUMN] == 'High Performance']
low_perf_df = df[df[TARGET_COLUMN] == 'Low Performance']

# 4. Print Summary
print(f"\nTotal samples analyzed: {len(df)}")
print(f"High Performance samples: {len(high_perf_df)}")
print(f"Low Performance samples: {len(low_perf_df)}")
if len(high_perf_df) < 30 or len(low_perf_df) < 30:
    print("\nWARNING: Your sample size is small. These insights are directional, not statistically significant.")
print("-" * 40)

# 5. Analyze Top Features

# --- Feature 1: Speaking Rate ---
print("\n--- 1. Optimal Speaking Rate (words per minute) ---")
if len(high_perf_df) > 0 and 'speaking_rate' in high_perf_df.columns:
    print(f"High Performance (Average): {high_perf_df['speaking_rate'].mean():.2f}")
else:
    print("High Performance (Average): N/A (no samples or column missing)")
if len(low_perf_df) > 0 and 'speaking_rate' in low_perf_df.columns:
    print(f"Low Performance (Average): {low_perf_df['speaking_rate'].mean():.2f}")
else:
    print("Low Performance (Average): N/A (no samples or column missing)")
print("Insight: A higher average speaking rate appears to correlate with 'High Performance' videos.")

# --- Feature 2: Punctuation Count ---
print("\n--- 2. Ideal Punctuation Marks (in caption) ---")
if len(high_perf_df) > 0 and 'cap_punctuation_count' in high_perf_df.columns:
    print(f"High Performance (Average): {high_perf_df['cap_punctuation_count'].mean():.2f}")
else:
    print("High Performance (Average): N/A (no samples or column missing)")
if len(low_perf_df) > 0 and 'cap_punctuation_count' in low_perf_df.columns:
    print(f"Low Performance (Average): {low_perf_df['cap_punctuation_count'].mean():.2f}")
else:
    print("Low Performance (Average): N/A (no samples or column missing)")
print("Insight: 'High Performance' videos seem to use fewer punctuation marks in their captions.")

# --- Feature 3: Month ---
print("\n--- 3. Best Month to Post ---")
print("\nDistribution of 'High Performance' posts by month:")
if len(high_perf_df) > 0 and 'month' in high_perf_df.columns:
    print(high_perf_df['month'].value_counts().sort_index().to_markdown())
else:
    print("N/A (no 'High Performance' samples or column missing)")

print("\nDistribution of 'Low Performance' posts by month:")
if len(low_perf_df) > 0 and 'month' in low_perf_df.columns:
    print(low_perf_df['month'].value_counts().sort_index().to_markdown())
else:
    print("N/A (no 'Low Performance' samples or column missing)")
print("Insight: Look for months where 'High' posts are common and 'Low' posts are rare (e.g., Summer vs. Fall).")