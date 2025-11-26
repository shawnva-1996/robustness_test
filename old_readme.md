feature engineering

i want to create a predictive model to predict the number of likes a video will get

I select a feature to predict. (log Likes) 

I extract all possible features. i drop all features that are engagement related and i drop all features that appear after a post is up ( (to avoid data leakage) )

From the first set of features, after extraction, I do feature engineering, run ridge/xgboost (or other models), do an analysis of feature importance and drop the least important features, 

add another set of features, do a second round of feature engineering, run ridge/xgboost (or other models), do an analysis of feature importance and drop the least important features,

I then add the third set of features relating to video information, do a 3rd round of feature engineering, run ridge/xgboost (or other models), do an analysis of feature importance and drop the least important features,

i then pick the final model and test it on a new video it has never seen before.

how do i know which model to select?

1. Merge the 60 videos from 6 buckets as discussed with Prof Steven
2. Drop the features that are not pre-post and create a log of likes column

2a. refer to the attached 2_processed_with_splits.csv. i want to create a predictive model that targets likes, should i use xgboost or a regression model?


3. do feature engineering on 2_processed.csv to produce 3_features_engineered.csv
4. after first pass of feature engineering. 

create a python script which will look at directory "CSV_with_video_data_extracted", with videoID as the primary key, merge the contents of the directory "CSV_with_video_data_extracted" with 3_processed_with_splits_full_engineered.csv and create 4_first_pass_with_video_data.csv. 

refer to the attached 3_processed_with_splits_full_engineered.csv and the contents of the directory "CSV_with_video_data_extracted" is the other attached csvs

5. remove data leakage caused by merging csv in 4

6. perform another round of feature engineering.

analyse the attached 5_data_without_leakage_features.csv and suggest features to engineer. my target is log likes so to avoid data leakage, do not engineer features that contain engagement metrics. suggest a complete summary of features to engineer and generate a python script called 6_video_features_engineered

7. obtain audio transcription, merge with 6, and then do another round of feature engineering

8. change target to predict whether a video will go viral, mid-viral, or not viral at all

9. change from high, mid, and low viral to  change target to high and low performance

Based on the entire data analysis and modeling journey you've completed, here is the clear, data-driven answer to what distinguishes viral from non-viral videos.

### The Single Biggest Factor: The Creator's History 👨‍💻

The most important and consistent finding across all your valid models is that **a video's potential for going viral is overwhelmingly determined by the creator's past performance, not the specific content of the video itself.**

Your most successful engineered features proved this:

* **`hearts_per_video` (Most Important):** This was a home run. It measures a creator's historical engagement rate. A high average suggests a loyal audience that consistently enjoys their content, making their next video far more likely to succeed.
* **`follower_video_ratio`:** This measures a creator's efficiency. Creators who gain a large following with fewer videos have likely found a successful formula, which your model correctly identifies as a strong positive signal.
* **`follower_to_following_ratio`:** This classic metric of influence also proved to be a significant factor.

**In layman's terms:** Your model learned that the best way to predict a video's future is to look at the creator's past. A video from a creator with a proven track record of high engagement is the safest bet for success.

---
### Secondary Factors: Video & Audio Content 🎬

While less important than the creator's history, some content-related features did emerge as secondary predictors. These are the "tie-breakers" that can make a difference, but they don't carry the same weight as the creator's profile.

* **Video Sharpness (`avg_sharpness`):** Higher sharpness, which often correlates with better production quality, was a consistent signal in your later models.
* **Creator's Socials (`Creator Following Count`):** How many other accounts the creator follows was surprisingly important. This could be a proxy for how engaged they are with the community or their level of "spam-like" activity.
* **Time of Post (`month`, `sin_hour`/`cos_hour`):** The time of year and day of the week a video is posted has a small but measurable impact on performance.
* **Transcript Word Count:** How "talkative" a video is played a minor role, suggesting that the pacing and density of information can influence engagement.

### What *Didn't* Matter as Much

It's equally important to note what *didn't* have a strong impact in the final, honest models:

* **Specific Topics:** Whether a video was about "Alexa" or was an "unboxing" (keyword flags or even the LLM topic) was not a top predictor.
* **Calls to Action (CTAs):** Explicitly asking for likes or follows in the audio didn't emerge as a top-tier feature.

### Final Conclusion

You've successfully answered your question. The data shows a clear hierarchy of what makes a video go viral:

1.  **Primary Driver:** A creator with a strong history of high-quality, engaging content (`hearts_per_video`).
2.  **Secondary Driver:** A creator with an efficient growth strategy (`follower_video_ratio`).
3.  **Tertiary Drivers:** Good production quality (`avg_sharpness`) and other content-specific factors.

In essence, your exhaustive analysis has proven the old adage: **"bet on the jockey, not just the horse."**


summarise Prof Steven's comment
1. first slide - summarise what i am doing
2. 2nd slide - summarise prof steven coments
2.1 - data driven feature engineering
- what is it?
- allow the data 
- principal component analysis (embeddings vertex)
2.2 - try my current approach on other people videos

3. https://www.kaggle.com/code/translator/google-gen-ai-capstone

3.1 convert this code to create embeddings of videos 
3.2 use these embeddings for data driven feature engineering
3.3 use kaggle

ask gemini if needed to revise this:
class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]# robustness_test
