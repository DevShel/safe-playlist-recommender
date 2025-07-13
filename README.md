# SafeSongs: A Safety-Aware Playlist Generator

## Overview

SafeSongs is a personalized recommendation system designed to generate music playlists that prioritize user safety by filtering out toxic content. It leverages user-specific safety thresholds and an XGBoost-based reranker to ensure playlists align with each listener's content preferences and sensitivity levels.

At a high-level, in this project I:
1. Determine a user's song listening preferences related to categories such as sexuality and violence.
2. Provide them song recommendations based on collaborative filtering via matrix factorization (users with similar listening habits liked these songs, so you probably will).
3. Rerank these intial collaborative filtering recommendations based on a user's safety preferences.

I built this project to explore practical applications of responsible AI.



### Data Sources
* **Echo Nest Taste Profile:** Contains user listening behaviors with play counts. This dataset also comes with the data from the Million Song Dataset, which provides song titles and artist names, indexed based on "song ID". Both datasets use the same "song ID" identifier. [(LINK)](https://www.kaggle.com/datasets/bivanmallick/million-song-recoomendation?select=triplets_file.csv)
* **Genius Lyrics Dataset:** Lyrics data for song content analysis. [(LINK)](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
* **OpenAI Moderation API:** Used to calculate toxicity scores across multiple content categories. [(LINK)](https://platform.openai.com/docs/guides/moderation?example=text#page-top)

## How It Works

### 1. Data Collection and Preparation

* Merged and cleaned datasets from Echo Nest Taste Profile, Million Song Dataset, and Genius to build comprehensive user-song interaction records.
  * Cleaning involved removed non-english songs, songs that don't have lyric data, etc..
* Computed toxicity scores for the lyrics of each song using OpenAI's Moderation API, generating detailed toxicity/safety profiles for each song. Used batching for increased efficiency. 
  * This API takes in song lyrics, and produces a JSON object with scores in range (0.00-1.00) for each of 9 respective categories, rating that song's lyrics in areas such as violence, sexuality, etc..


### 2. User Toxicity Threshold Calculation
* Calculated an upper bound for each user's listening preferences, by taking the weighted 85th percentile of songs they have listened to, in each of the 9 safety categories (violence, sexual, etc..), weighted by number of listens.
* Then took the unweighted 85th percentile of all of these values across all users, to set default upper-bound values for users with less than 5 song listens, to combat cold-start problems. I picked 5 songs as a cutoff, because I believe that users with less than 5 songs don't have enough activity to make a judgement about their safety preferences.


### 3. Song Recommendation Generation
* Initial candidate playlist generated using collaborative filtering, via the Surprise library’s Singular Value Decomposition (SVD) matrix-factorization algorithm, which learns latent user- and item-factors from the play-count matrix.


### 4. XGBoost Reranking
* The XGBoost Regressor reranker takes in a few features including the initial collaborative filtering score, distance between user preference and song safety value which is the following subtraction: (user's weighted 85th percentile upper bound - this current song's safety score) in each of the 9 categories, and finally the user's average listens per song (to demonstrate how explorative they are).
* The reranker then predicts log(play_count), which is the log-normalized playcount of the song, normalized in training and at inference time due to some users listening to songs a lot.
* I opted to use a regressor because this regressor made more intuitive sense to me, but there is also a learning-to-rank capability of XGBoost.
* This XGBoost Regressor learns to penalize songs that exceed personalized toxicity thresholds across multiple categories (violence, harassment, explicit content, etc.). This ensures recommended songs align closely with each user's demonstrated preferences.
* Given an initial recommendation set of 250 songs, a playlist can now be generated using the top 50 or 100 candidates, that will include songs optimized for user satisfaction which align with their preferences in these safety-related areas.

## Results
When looking at the bottom of the Recommendation_System.ipynb file, you can see that songs, which would originally rank higher if we solely relied on the SVD matrix factorization, rank lower, due to them being misaligned with the user's traditional listening preferences. This shows the usefulness of the XGBoost reranking step!