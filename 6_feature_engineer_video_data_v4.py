import pandas as pd
import numpy as np
import emoji

# --- Configuration ---
INPUT_FILENAME = '5_data_without_leakage_features.csv'
OUTPUT_FILENAME = '6_feature_engineer_video_data.csv'
# ---------------------

print(f"Loading data from '{INPUT_FILENAME}'...")
df = pd.read_csv(INPUT_FILENAME)

# --- 1. Engineer Creator-Based Ratio Features ---
# print("Engineering creator-based features...")
# --- LEAKAGE REMOVED ---
# df['follower_to_following_ratio'] = df['Creator Follower Count'] / df['Creator Following Count'].replace(0, 1)
# df['hearts_per_video'] = df['Creator Total Heart Count'] / df['Creator Total Video Count'].replace(0, 1)


# --- 2. Engineer Advanced Time-Based Features ---
print("Engineering time-based features...")
# Ensure timestamp is in datetime format
# --- FIX: Changed '_parsed_time' to 'Post Timestamp' ---
df['Post Timestamp'] = pd.to_datetime(df['Post Timestamp'])

# Day of the week (Monday=0, Sunday=6)
# --- FIX: Changed '_parsed_time' to 'Post Timestamp' ---
df['day_of_week'] = df['Post Timestamp'].dt.dayofweek

# Time of day buckets
df['time_of_day_bucket'] = pd.cut(
    df['hour_of_day'],
    bins=[-1, 5, 12, 17, 21, 24],
    labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']
)
# "Golden Hour" / Prime Time flag
df['is_prime_time'] = df['hour_of_day'].between(18, 22, inclusive='both').astype(int)


# --- 3. Engineer Advanced Caption-Based Features ---
print("Engineering caption-based features...")
# Fill missing captions with an empty string for safe processing
df['Post Caption'] = df['Post Caption'].fillna('')

# Emoji count
df['emoji_count'] = df['Post Caption'].apply(emoji.emoji_count)

# Uppercase word ratio
def uppercase_ratio(text):
    words = text.split()
    if not words:
        return 0
    # Count words that are fully uppercase and longer than 1 character
    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    return uppercase_words / len(words)

df['uppercase_ratio'] = df['Post Caption'].apply(uppercase_ratio)

# Question-asking (anywhere in caption)
df['has_question'] = df['Post Caption'].str.contains('?', regex=False).astype(int)


# --- 4. Engineer Interaction Features ---
print("Engineering interaction features...")

# Creator Influence Score
# --- LEAKAGE REMOVED ---
# df['creator_influence_score'] = df['log1p_followers'] * df['is_verified_int']

# "Busy" Video Score
df['busy_video_score'] = df['avg_motion'] * df['cuts_per_second']

# Hashtag Density
df['hashtag_density'] = df['hashtag_count'] / df['caption_length'].replace(0, 1)


# --- Save the new dataset ---
df.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Successfully engineered all features and saved to '{OUTPUT_FILENAME}'")

# --- UPDATED: List of only the new, safe features ---
new_features = [
    'day_of_week', 'time_of_day_bucket', 'is_prime_time', 'emoji_count',
    'uppercase_ratio', 'has_question', 'busy_video_score', 'hashtag_density'
]
print("\nNew features added:")
for feature in new_features:
    print(f"- {feature}")