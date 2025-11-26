import pandas as pd

# --- Configuration ---
INPUT_METADATA = '6_feature_engineer_video_data.csv'
INPUT_AI_FEATURES = '7c_final_features.csv'

# We are outputting to this specific name so that your existing '7_train_model.py'
# script will run without any changes.
OUTPUT_FINAL_DATASET = '7b_data_with_full_transcript_features.csv' 

# ---------------------

print("--- Starting Final Dataset Merge ---")

# --- 1. Load your clean, non-leaky metadata ---
print(f"Loading non-leaky metadata from '{INPUT_METADATA}'...")
try:
    metadata_df = pd.read_csv(INPUT_METADATA)
except FileNotFoundError:
    print(f"Error: File not found: '{INPUT_METADATA}'")
    print("Please run the corrected '6_feature_engineer_video_data_v4.py' first.")
    exit()

# --- 2. Load your clean, non-leaky AI features ---
print(f"Loading non-leaky AI features from '{INPUT_AI_FEATURES}'...")
try:
    ai_features_df = pd.read_csv(INPUT_AI_FEATURES)
except FileNotFoundError:
    print(f"Error: File not found: '{INPUT_AI_FEATURES}'")
    print("Please run the '7c_merge_llm_topic.py' script first.")
    exit()

# --- 3. Ensure Video ID is a string for a reliable merge ---
metadata_df['Video ID'] = metadata_df['Video ID'].astype(str)
ai_features_df['Video ID'] = ai_features_df['Video ID'].astype(str)

# --- 4. Merge the datasets ---
print("Merging metadata and AI features on 'Video ID'...")

# We use a 'left' merge to keep every row from your main metadata file
# and join the matching AI features to it.
final_df = pd.merge(metadata_df, ai_features_df, on='Video ID', how='left')

# --- 5. Save the final dataset ---
final_df.to_csv(OUTPUT_FINAL_DATASET, index=False)

print("\n✅ Success! Your true final dataset is ready.")
print(f"File saved as '{OUTPUT_FINAL_DATASET}' with shape {final_df.shape}")