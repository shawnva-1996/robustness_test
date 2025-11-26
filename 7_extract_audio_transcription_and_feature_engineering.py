import os
import pandas as pd
import whisper
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    vfx,
    concatenate_videoclips,
)
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import google.generativeai as genai # <-- IMPORT NEW LIBRARY
import re # <-- Import regex for cleaning

# --- Configuration ---
VIDEO_DIRECTORY = 'videos'
INPUT_CSV = '6_feature_engineer_video_data.csv'
INTERMEDIATE_CSV = '7a_transcripts_and_base_features.csv'
FINAL_OUTPUT_CSV = '7b_data_with_full_transcript_features.csv'
WHISPER_MODEL = 'base'

# # --- Llama 3.1 Configuration ---
# OLLAMA_ENDPOINT = 'http://localhost:11434/api/generate'
# LLAMA_MODEL = 'llama3.1:8b'
# MAX_LLM_WORKERS = 4 # Adjust based on your Mac's performance (M4 Air can likely handle 4-6)

# --- Gemini Configuration ---
# Set your API key in your environment variables for security
# (e.g., export GEMINI_API_KEY='your_key_here')
try:
    genai.configure(api_key="AIzaSyCRYwTZ-xooWz9TA6g25746IIFzt1isjkE")
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit()

GEMINI_MODEL = 'gemini-2.5-pro' #'gemini-1.5-pro-latest' # Or 'gemini-pro'

# Gemini 2.5 Pro	gemini-2.5-pro	Top-tier performance. Best for complex reasoning, coding, and challenging creative tasks.
# Gemini 2.5 Flash	gemini-2.5-flash	Speed & Performance. The best balance of speed and capability, great for chat, summarization, and high-frequency tasks.
# Gemini 2.5 Flash-Lite	gemini-2.5-flash-lite	Fastest / Most Cost-Efficient. Ideal for highly scaled, low-latency tasks where cost is a major factor.
# Gemini 2.0 Flash	gemini-2.0-flash	A previous-generation fast model, still available and reliable.

MAX_LLM_WORKERS = 4 # You can increase this with an API

# --- Feature Engineering Keyword Lists ---
CTA_KEYWORDS = ["link in bio", "follow for more", "subscribe", "download", "comment below"]
TOPIC_KEYWORDS = ["alexa", "google home", "review", "unboxing", "tutorial", "how to", "setup", "automation"]
# ---------------------

# def get_topic_from_llama(transcript):
#     """Sends a single transcript to the local Ollama model."""
#     prompt = f"""
#     Read the following video transcript. Classify the main topic of the video into ONE of the following categories:
#     - Product Review
#     - Tutorial/How-To
#     - News/Update
#     - Product Comparison
#     - General Discussion
#     - Smart Home Tour

#     Respond with ONLY the category name and nothing else.

#     Transcript: "{transcript}"
#     """
#     payload = {"model": LLAMA_MODEL, "prompt": prompt, "stream": False}
#     try:
#         response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
#         response.raise_for_status()
#         return response.json().get('response', 'Topic_Error').strip()
#     except requests.exceptions.RequestException as e:
#         print(f"Llama API Error: {e}")
#         return "Topic_Unavailable"

def get_features_from_gemini(transcript):
    """
    Sends a single transcript to the Gemini API for advanced feature extraction.
    We are asking for a JSON object for reliable parsing.
    """
    
    # This is a much more powerful prompt than the Llama one
    prompt = f"""
    Analyze the following video transcript. Return a JSON object with the following keys:
    - "topic": Classify the main topic into ONE of: [Product Review, Tutorial/How-To, News/Update, Product Comparison, General Discussion, Smart Home Tour, Unboxing, Other]
    - "sentiment": Classify the overall sentiment as: [Positive, Negative, Neutral]
    - "tone": Classify the primary tone as: [Informational, Funny, Serious, Inspirational, Sales-focused]
    - "summary": Provide a concise 1-sentence summary of the video.

    Transcript:
    "{transcript}"

    Respond ONLY with the raw JSON object.
    """
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        # Clean the response to get just the JSON
        json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        features = json.loads(json_text)
        
        # Standardize for merging
        return {
            'llm_topic': features.get('topic', 'Topic_Error'),
            'llm_sentiment': features.get('sentiment', 'Sentiment_Error'),
            'llm_tone': features.get('tone', 'Tone_Error'),
            'llm_summary': features.get('summary', 'Summary_Error')
        }
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {
            'llm_topic': 'Topic_Unavailable',
            'llm_sentiment': 'Sentiment_Unavailable',
            'llm_tone': 'Tone_Unavailable',
            'llm_summary': 'Summary_Unavailable'
        }

def process_video_file(video_path, whisper_model):
    """Processes a single video file to get duration, transcript, and non-LLM features."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    with VideoFileClip(video_path) as video:
        duration = video.duration
        
    result = whisper_model.transcribe(video_path, fp16=False)
    
    transcript = result['text']
    text = transcript.lower() if isinstance(transcript, str) else ''
    
    word_count = len(text.split())
    speaking_rate = word_count / duration if duration > 0 else 0
    question_count = text.count('?')
    has_cta = 1 if any(phrase in text for phrase in CTA_KEYWORDS) else 0
    
    features = {
        'Video ID': video_id,
        'transcript': transcript,
        'transcript_word_count': word_count,
        'speaking_rate': speaking_rate,
        'transcript_question_count': question_count,
        'transcript_has_cta': has_cta,
    }
    
    for keyword in TOPIC_KEYWORDS:
        features[f'keyword_{keyword.replace(" ", "_")}'] = 1 if keyword in text else 0
        
    return features

# --- Main Script ---
if __name__ == "__main__":
    # === STAGE 1: Transcribe and Pre-process in a Batch ===
    print("--- Stage 1: Processing all video files ---")
    
    if not os.path.isdir(VIDEO_DIRECTORY):
        print(f"Error: Directory '{VIDEO_DIRECTORY}' not found.")
        exit()
        
    video_files = [os.path.join(VIDEO_DIRECTORY, f) for f in os.listdir(VIDEO_DIRECTORY) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No .mp4 files found in '{VIDEO_DIRECTORY}'.")
        exit()
        
    print(f"Loading Whisper model ({WHISPER_MODEL})...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    
    all_video_data = []
    # Process videos with a progress bar
    with tqdm(total=len(video_files), desc="Transcribing Videos") as pbar:
        for video_path in video_files:
            try:
                all_video_data.append(process_video_file(video_path, whisper_model))
            except Exception as e:
                print(f"Error processing {os.path.basename(video_path)}: {e}")
            pbar.update(1)

    if not all_video_data:
        print("\nNo videos were processed successfully in Stage 1. Exiting.")
        exit()

    intermediate_df = pd.DataFrame(all_video_data)
    intermediate_df.to_csv(INTERMEDIATE_CSV, index=False)
    print(f"\nStage 1 complete. Intermediate data saved to '{INTERMEDIATE_CSV}'.")
    
    # Load intermediate data
    intermediate_df = pd.read_csv(INTERMEDIATE_CSV)

    # === STAGE 2: Classify Topics with LLM in Parallel ===
    print(f"\n--- Stage 2: Classifying topics in parallel using {GEMINI_MODEL} ---")
    
    transcripts_to_process = intermediate_df['transcript'].fillna("").tolist()
    all_llm_features = []
    
    with ThreadPoolExecutor(max_workers=MAX_LLM_WORKERS) as executor:
        future_to_index = {executor.submit(get_features_from_gemini, transcript): i for i, transcript in enumerate(transcripts_to_process)}
        
        results = [None] * len(transcripts_to_process)

        with tqdm(total=len(transcripts_to_process), desc="Classifying Topics") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error classifying transcript at index {index}: {e}")
                    results[index] = {
                        'llm_topic': 'Topic_Unavailable',
                        'llm_sentiment': 'Sentiment_Unavailable',
                        'llm_tone': 'Tone_Unavailable',
                        'llm_summary': 'Summary_Unavailable'
                    }
                pbar.update(1)

    # Convert the list of dictionaries into a DataFrame
    llm_features_df = pd.DataFrame(results)
    
    # Combine with the intermediate_df
    intermediate_df = pd.concat([intermediate_df, llm_features_df], axis=1)
    
    # --- Final Merge ---
    print("\nTopic classification complete. Merging all data...")
    
    main_df = pd.read_csv(INPUT_CSV)
    
    main_df['Video ID'] = main_df['Video ID'].astype(str)
    intermediate_df['Video ID'] = intermediate_df['Video ID'].astype(str)
    
    # Drop the now-redundant transcript column before merging
    intermediate_df = intermediate_df.drop(columns=['transcript'], errors='ignore')
    
    final_df = pd.merge(main_df, intermediate_df, on='Video ID', how='left')
    
    final_df.to_csv(FINAL_OUTPUT_CSV, index=False)
    
    print(f"\n✅ Success! Final dataset saved to '{FINAL_OUTPUT_CSV}'.")
    print(f"The new file has {final_df.shape[0]} rows and {final_df.shape[1]} columns.")