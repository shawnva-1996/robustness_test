import pandas as pd
import numpy as np

def process_data(input_filename="1_merged.csv", output_filename="2_processed.csv"):
    """
    Removes specified columns and adds a log-transformed 'Like Count' column.

    Args:
        input_filename (str): The name of the input CSV file.
        output_filename (str): The name of the file for the processed CSV output.
    """
    try:
        # Load the dataset
        print(f"Reading data from '{input_filename}'...")
        df = pd.read_csv(input_filename)

        # A list of columns to be removed
        features_to_remove = [
            'Comment Count',
            'Share Count',
            'Save Count',
            'Repost Count',
            'Video Codec'
            # --- NEW: ADD LEAKY CREATOR FEATURES ---
            'Creator Follower Count',
            'Creator Following Count',
            'Creator Total Heart Count',
            'Creator Total Video Count'
        ]

        # We'll check which of these columns actually exist in the DataFrame to avoid errors
        existing_features_to_remove = [col for col in features_to_remove if col in df.columns]
        
        # Remove the specified features
        df_processed = df.drop(columns=existing_features_to_remove)
        print(f"Successfully removed columns: {existing_features_to_remove}")

        # --- Create a log of the 'Like Count' column ---
        if 'Like Count' in df_processed.columns:
            # We add 1 to the like count to prevent an error from log(0)
            df_processed['log_likes'] = np.log(df_processed['Like Count'] + 1)
            print("Successfully created the 'log_likes' column.")
        else:
            print("Warning: 'Like Count' column not found. Could not create 'log_likes'.")

        # Save the processed dataframe to a new CSV file
        df_processed.to_csv(output_filename, index=False)
        print(f"\nProcessing complete! The new file is saved as '{output_filename}'")
        
        print("\nHere's a preview of the first 5 rows of your new data:")
        print(df_processed.head())

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        print("Please make sure the script is in the same directory as your CSV file.")

if __name__ == "__main__":
    process_data()