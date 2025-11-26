import pandas as pd

# --- Configuration ---
INPUT_BASE_FEATURES = '7a_transcripts_and_base_features.csv'
INPUT_TOPIC_SOURCE = '7b_data_with_full_transcript_features.csv'
OUTPUT_FINAL_DATA = '7c_final_features.csv'
# ---------------------

print(f"Loading base AI features from '{INPUT_BASE_FEATURES}'...")
try:
    base_df = pd.read_csv(INPUT_BASE_FEATURES)
except FileNotFoundError:
    print(f"Error: File not found: '{INPUT_BASE_FEATURES}'")
    exit()

print(f"Loading new 'llm_topic' data from '{INPUT_TOPIC_SOURCE}'...")
try:
    topic_df = pd.read_csv(INPUT_TOPIC_SOURCE)
except FileNotFoundError:
    print(f"Error: File not found: '{INPUT_TOPIC_SOURCE}'")
    exit()

# --- Select only the 'llm_topic' and 'Video ID' from the source file ---
llm_topic_data = topic_df[['Video ID', 'llm_topic']]

# --- Merge the datasets ---
print("Merging 'llm_topic' into the base feature set...")

# Ensure Video ID is the same type for merging
base_df['Video ID'] = base_df['Video ID'].astype(str)
llm_topic_data['Video ID'] = llm_topic_data['Video ID'].astype(str)

# --- Check if 'llm_topic' already exists in the base file ---
if 'llm_topic' in base_df.columns:
    print("Found 'llm_topic' in the base file. Dropping it to replace with the new one.")
    base_df = base_df.drop(columns=['llm_topic'])

final_df = pd.merge(base_df, llm_topic_data, on='Video ID', how='left')

# Save the final dataset
final_df.to_csv(OUTPUT_FINAL_DATA, index=False)

print(f"\n✅ Success! Your final feature set is ready.")
print(f"File saved as '{OUTPUT_FINAL_DATA}' with shape {final_df.shape}")