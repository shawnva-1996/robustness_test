# Robustness Test & Validation Suite

## 📋 Project Overview
This software suite serves as the validation component for the **Search Video Optimisation (SVO)** project (https://github.com/shawnva-1996/svo-va). While the main SVO app is designed for client-specific data, this suite stress-tests the machine learning pipeline against an **external dataset** of 60 videos (30 International, 30 Singaporean).

**Key Capabilities:**
* **Data Leakage Detection:** Scripts specifically designed to strip post-event metrics (e.g., Play Counts) to ensure model integrity.
* **Multimodal AI Analysis:** Uses **OpenAI Whisper** (local) for audio transcription and **Google Gemini 2.5 Pro** (API) for semantic topic/tone analysis.
* **Generalization Testing:** Evaluates model performance on unseen data, demonstrating the pivot from Regression (predicting likes) to Classification (predicting performance tiers).

## ⚙️ System Prerequisites (Critical)

Before running any Python code, you **must** install these system-level tools. The audio processing scripts will fail without them.

### 1. Install FFmpeg
Required for `whisper` and `moviepy` to process video audio tracks.

**Windows:**
1.  Download the "essentials" build from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
2.  Extract the ZIP file.
3.  Rename the extracted folder to `ffmpeg` and move it to the `C:\` drive (e.g., `C:\ffmpeg`).
4.  Add the `bin` folder to your System Path:
    * Press `Win + R`, type `sysdm.cpl`, and hit Enter.
    * Go to **Advanced** tab > **Environment Variables**.
    * Under "System variables", find **Path** and click **Edit**.
    * Click **New** and paste: `C:\ffmpeg\bin`
    * Click OK on all windows.
5.  **Verify:** Open a new Command Prompt and type `ffmpeg -version`.

**Mac (Apple Silicon/Intel):**
1.  Open Terminal.
2.  Install via Homebrew (if you don't have Homebrew, install it from [brew.sh](https://brew.sh)):
    ```bash
    brew install ffmpeg
    ```

### 2. Google Gemini API Key
This project uses Google Gemini 2.5 Pro for semantic analysis.
1.  Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2.  You will need to paste this key into the script `7_extract_audio_transcription_and_feature_engineering.py` (line 27) OR set it as an environment variable.

---

## 🚀 Installation Guide

### Step 1: Project Setup
Create a folder for the project and move all the provided Python scripts (`.py`) into it.

Inside that folder, create two sub-folders for your data:
* `videos/` (Place your 60 .mp4 video files here)
* `CSV_to_be_merged/` (Place your raw .csv metadata files here)

### Step 2: Python Environment
Open your Terminal (Mac) or Command Prompt (Windows) in the project folder.

**Windows:**
```bash
python -m venv venv_robust
.\venv_robust\Scripts\activate
````

**Mac:**

```bash
python3 -m venv venv_robust
source venv_robust/bin/activate
```

### Step 3: Install Python Dependencies

Run the following command to install all required libraries for Machine Learning, AI API access, and Media processing.

```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn google-generativeai openai-whisper moviepy emoji tqdm requests
```

-----

## 🧪 How to Re-create the Pipeline (Step-by-Step)

Run these scripts in the exact order below to replicate the full analysis narrative: **Data Prep** $\to$ **Regression Failure** $\to$ **Classification Success**.

### Phase 1: Data Preparation

**1. Merge Raw Data**
Combines individual CSVs from the `CSV_to_be_merged` folder into one master file.

```bash
python 1_csv_merger.py
```

**2. Clean & Split Data**
Drops initial metadata and creates a **Time-Based** Train/Test split. This is crucial for preventing data leakage (training on future data to predict the past).

```bash
python 2_drop_pre_post_features_v2.py
python 2b_create_splits.py
```

### Phase 2: The "Failed" Regression Test

**3. Basic Feature Engineering**
Creates basic features to test if we can predict exact likes.

```bash
python 3_features_engineered_v2.py
```

**4. Train Regression Models**
*Observation:* This will output negative $R^2$ scores (e.g., -10.0), demonstrating that the regression approach fails to generalize on external data.

```bash
python 2_train_models_v2.py
```

### Phase 3: The "Rebuild" & Classification Pivot

**5. Remove Leakage**
Strictly removes features like `Play Count` and `Follower Count` that caused overfitting in earlier versions.

```bash
python 5_remove_data_leakage.py
```

**6. Advanced Feature Engineering**
Extracts "Safe" features like Emoji count, Uppercase ratio (shouting), and Time buckets.

```bash
python 6_feature_engineer_video_data_v4.py
```

**7. AI Semantic Extraction (The Heavy Lifter)**
*Note: This step may take 5-10 minutes depending on video length.*
It uses Whisper (local) to transcribe audio and Gemini (API) to extract Tone and Topic.

```bash
python 7_extract_audio_transcription_and_feature_engineering.py
```

**8. Create Target Variable**
Pivots the problem from predicting numbers to predicting categories ("High Performance" vs "Low Performance").

```bash
python 9_high_vs_low_perf.py
```

**9. Train Classification Model (The Success)**
Trains a Random Forest Classifier.
*Observation:* Check the **Recall** score. It should be high (1.00) for the target class, indicating it successfully flags potential hits.

```bash
python 9_train_model.py
```

**10. Generate Insights**
Prints the final analysis of what drives performance (e.g., Speaking Rate, Seasonality).

```bash
python 10_analyse_insights.py
```

-----

## 📂 Troubleshooting

  * **"ffmpeg not found" or "FileNotFoundError: [WinError 2]":**
      * This means FFmpeg is not installed or not in your PATH. Refer to the "System Prerequisites" section above. You may need to restart your computer after adding it to the PATH.
  * **"google.generativeai.types.generation\_types.StopCandidateException":**
      * This usually means the AI detected safety concerns in a video transcript. The script handles this gracefully, but ensure your API key is valid.
  * **"IndexError: list index out of range" in script 1:**
      * Ensure your `CSV_to_be_merged` folder is not empty and contains valid CSV files.
