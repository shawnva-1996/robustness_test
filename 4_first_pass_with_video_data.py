import pandas as pd
import glob
import os

# --- Configuration ---
# Folder containing the CSV files to be merged
video_data_folder = 'CSV_with_video_data_extracted'

# The main CSV file to merge with
main_file = '3_processed_with_splits_full_engineered.csv'

# Name of the final output file
output_filename = '4_first_pass_with_video_data.csv'
# ---------------------

# Check if the target folder exists
if not os.path.isdir(video_data_folder):
    print(f"Error: The folder '{video_data_folder}' was not found.")
    print("Please make sure the folder exists and is in the same directory as the script.")
else:
    # Create the full path to search for CSV files inside the folder
    search_path = os.path.join(video_data_folder, '*.csv')
    video_data_files = glob.glob(search_path)

    if not video_data_files:
        print(f"No CSV files were found in the '{video_data_folder}' directory.")
    else:
        try:
            print(f"Found {len(video_data_files)} CSV files to merge.")
            
            # Load and concatenate all video data CSVs from the folder
            video_data_df_list = [pd.read_csv(file) for file in video_data_files]
            video_data_df = pd.concat(video_data_df_list, ignore_index=True)

            # Remove any duplicate rows based on the 'Video ID'
            video_data_df.drop_duplicates(subset='Video ID', keep='first', inplace=True)

            # Load the main processed data file
            processed_df = pd.read_csv(main_file)

            # Identify new columns to add from the video data
            # (all columns from video data except those already in the main file, plus 'Video ID' for the merge)
            processed_columns = processed_df.columns.tolist()
            video_data_columns = video_data_df.columns.tolist()
            columns_to_merge = ['Video ID'] + [col for col in video_data_columns if col not in processed_columns]

            # Create a smaller dataframe with only the new columns to merge
            video_features_to_merge = video_data_df[columns_to_merge]

            # Merge the main dataframe with the new video features
            # 'how='left'' ensures that all rows from the main file are kept
            merged_df = pd.merge(processed_df, video_features_to_merge, on='Video ID', how='left')

            # Save the final merged dataframe to a new CSV file
            merged_df.to_csv(output_filename, index=False)

            print(f"\n✅ Success! Merged the files and created '{output_filename}'")
            print(f"Shape of the final dataframe: {merged_df.shape}")

        except FileNotFoundError:
            print(f"\nError: The main file '{main_file}' was not found.")
            print("Please make sure it's in the same directory as the script.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")