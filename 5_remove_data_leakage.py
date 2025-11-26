import pandas as pd

# --- Configuration ---
input_filename = '4_first_pass_with_video_data.csv'
output_filename = '5_data_without_leakage_features.csv'

# Define the list of features (columns) to remove.
# These are often removed to prevent 'data leakage', where the model is trained
# on information that it wouldn't have at the time of prediction.
features_to_remove = [
    'Play Count',
    'Comment Count',
    'Share Count',
    'Save Count',
    'Repost Count',
    'Video Codec'
]
# ---------------------

try:
    print(f"Reading data from '{input_filename}'...")
    df = pd.read_csv(input_filename)
    
    # Get the original number of columns for comparison
    original_column_count = df.shape[1]
    
    print(f"\nAttempting to remove the following {len(features_to_remove)} features:")
    for feature in features_to_remove:
        print(f"- {feature}")

    # Drop the specified columns from the dataframe
    # The 'errors='ignore'' flag prevents an error if a column doesn't exist
    df_cleaned = df.drop(columns=features_to_remove, errors='ignore')
    
    cleaned_column_count = df_cleaned.shape[1]

    print(f"\nSuccessfully removed {original_column_count - cleaned_column_count} feature(s).")
    
    # Save the cleaned dataframe to a new CSV file
    df_cleaned.to_csv(output_filename, index=False)
    
    print(f"✅ Cleaned data has been saved to '{output_filename}'")
    print(f"The new file has {df_cleaned.shape[0]} rows and {cleaned_column_count} columns.")

except FileNotFoundError:
    print(f"\nError: The file '{input_filename}' was not found.")
    print("Please make sure the script is in the same directory as your CSV file.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")