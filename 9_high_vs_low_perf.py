import pandas as pd

# --- Configuration ---
INPUT_FILENAME = '7b_data_with_full_transcript_features.csv'
OUTPUT_FILENAME = '8_data_with_performance_category.csv'
# ---------------------

print(f"Loading data from '{INPUT_FILENAME}'...")
try:
    df = pd.read_csv(INPUT_FILENAME)
    print("Successfully loaded file.")
except FileNotFoundError:
    print(f"Error: The file '{INPUT_FILENAME}' was not found.")
    exit()

# Define the mapping from the old categories to the new, simpler ones
category_map = {
    'viral': 'High Performance',
    'mid-viral': 'High Performance',
    'non-viral': 'Low Performance'
}

# --- Create the new 'performance_category' column ---
# The .map() function will apply our dictionary to the 'viral' column
df['performance_category'] = df['viral'].map(category_map)

print("\nTransformation complete. Here is a summary:")

# Show the value counts of the original 'viral' column
print("\nOriginal 'viral' column distribution:")
print(df['viral'].value_counts())

# Show the value counts of the new 'performance_category' column
print("\nNew 'performance_category' column distribution:")
print(df['performance_category'].value_counts())


# --- Save the new dataset ---
df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Successfully created the new target column and saved the dataset to '{OUTPUT_FILENAME}'")