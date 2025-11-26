import pandas as pd
import os

def merge_csv_files_with_features(directory_path="CSV_to_be_merged", output_filename="1_merged.csv"):
    """
    Scans a directory for CSV files, merges them, and adds 'Region Flag' and 'viral'
    features based on the filenames.

    Args:
        directory_path (str): The name of the directory containing the CSV files.
        output_filename (str): The name of the file for the merged CSV output.
    """
    dataframes = []
    
    print(f"Scanning for CSV files in '{directory_path}'...")

    try:
        files_in_directory = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' was not found.")
        print("Please make sure the directory exists and is correctly named.")
        return

    for filename in files_in_directory:
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # --- Feature Creation ---
            # Remove file extension and split by underscore
            parts = filename[:-4].split('_')

            # Determine Region Flag
            if 'global' in parts:
                df['Region Flag'] = 'Global'
            elif 'sg' in parts:
                df['Region Flag'] = 'SG'
            else:
                df['Region Flag'] = 'Unknown'

            # Determine Viral Status
            if 'non' in parts:
                df['viral'] = 'non-viral'
            elif 'mid' in parts:
                df['viral'] = 'mid-viral'
            elif 'viral' in parts:
                df['viral'] = 'viral'
            else:
                df['viral'] = 'unknown'
            
            dataframes.append(df)
            print(f"Processed: {filename}")

    if not dataframes:
        print("No CSV files were found in the directory.")
        return

    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Save the final merged dataframe to a CSV file
    merged_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully merged all CSV files into '{output_filename}'.")

if __name__ == "__main__":
    merge_csv_files_with_features()