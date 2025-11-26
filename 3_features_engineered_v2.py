import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('2_processed_with_splits.csv')

# --- Preparatory Step: Convert Timestamp to Datetime ---
df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'])

# ==================================
# 1. Transforms & Rate Features (Non-Leaky)
# ==================================
# df['log1p_followers'] = np.log1p(df['Creator Follower Count'])
# df['log1p_total_videos'] = np.log1p(df['Creator Total Video Count'])
df['log1p_duration_s'] = np.log1p(df['Video Duration (s)'])
df['log1p_play_count'] = np.log1p(df['Play Count']) # Non-leaky log-transform of Play Count

# Creator Follower-to-Video Ratio (measure of creator efficiency)
# df['follower_video_ratio'] = np.where(df['Creator Total Video Count'] > 0, 
#                                      df['Creator Follower Count'] / df['Creator Total Video Count'], 
#                                      0)

# Video Dimension Aspect Ratio
def calculate_aspect_ratio(dim_str):
    if pd.isna(dim_str): return np.nan
    try:
        width, height = map(int, dim_str.split('x'))
        return width / height
    except ValueError:
        return np.nan

df['aspect_ratio'] = df['Video Dimensions'].apply(calculate_aspect_ratio)

# ==================================
# 2. Text Features (from Post Caption)
# ==================================
# Caption Length
df['caption_length'] = df['Post Caption'].str.len()

# Hashtag Count
df['hashtag_count'] = df['Post Caption'].str.count('#')

# Mention Count
df['mention_count'] = df['Post Caption'].str.count('@')

# Caption ends with a question mark
df['cap_ends_with_q'] = df['Post Caption'].str.strip().str.endswith('?').astype(int)

# Check for common Call To Action (CTA) keywords
cta_keywords = ['link in bio', 'tap', 'comment', 'follow', 'share', 'tag a friend', 'check out']
cta_pattern = r'|'.join(cta_keywords)
df['has_cta'] = df['Post Caption'].str.lower().str.contains(cta_pattern).astype(int)

# Total common punctuation count
df['cap_punctuation_count'] = df['Post Caption'].str.count(r'[!?.,]')


# ==================================
# 3. Time Buckets and Cyclical Features
# ==================================
df['hour_of_day'] = df['Post Timestamp'].dt.hour
df['day_of_month'] = df['Post Timestamp'].dt.day
df['month'] = df['Post Timestamp'].dt.month

# Hour-of-day Bins 
conditions = [
    (df['hour_of_day'] >= 5) & (df['hour_of_day'] <= 9),
    (df['hour_of_day'] >= 10) & (df['hour_of_day'] <= 16),
    (df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 21),
]
choices = ['commute', 'work', 'evening']
df['hour_bucket'] = np.select(conditions, choices, default='late')

# Weekday vs Weekend (1 for Sat/Sun, 0 for Mon-Fri)
df['is_weekend'] = (df['Post Timestamp'].dt.dayofweek >= 5).astype(int)

# Cyclical Hour Features (safe way to encode time)
df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

# ==================================
# 4. Metadata/Categorical Features
# ==================================
# Boolean to Integer conversion
df['is_verified_int'] = df['Creator Verified'].astype(int)
df['is_original_sound_int'] = df['Is Original Sound'].astype(int)

# Frequency Encoding for Music ID 
music_freq_map = df['Music ID'].value_counts().to_dict()
df['music_id_freq_enc'] = df['Music ID'].map(music_freq_map)

# Label Encoding for Video Definition
df['video_def_enc'] = df['Video Definition'].astype('category').cat.codes

# ==================================
# 5. Keyword Features
# ==================================
# Number of keywords listed
df['explicit_keyword_count'] = df['Keywords'].str.split(',').str.len().fillna(0).astype(int)

# Binary feature for presence of any keywords
df['has_keywords'] = (~df['Keywords'].isna()).astype(int)

# Save the fully transformed dataframe
df.to_csv('3_processed_with_splits_full_engineered.csv', index=False)