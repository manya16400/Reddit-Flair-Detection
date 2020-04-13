# Reddit-Flair-Detection
  A Reddit Flair Detector web application to detect flairs of India subreddit posts using Machine Learning algorithms and NLP.

## Table of Contents-
1. [About](https://github.com/manya16400/Reddit-Flair-Detection/edit/master/README.md/About/)
2. [Installation](/installation/)
3. [Data](/data/)
4. [Flair Classifier](/flair_classifier/)
5. [Deploying a Web app](/web_app/)
6. [References](/referneces/)
      
## About
  This repository illustrates the process of scraping reddit posts from the subreddit [r/india](https://www.reddit.com/r/india),    building a classifier to classify the posts into 8 different flairs and deploying the best model as a web application.

## Installation
  Along with Python 3, this project requires following libraries (a few others which are mentioned in later)  :
  * sklearn
  * Pytorch
  * pandas
  * numpy
  * matplotlib
  * nltk
  * praw
  
## Data
  To download the dataset
  <https://drive.google.com/open?id=1V2DaOj97SjZMTQa_MeJIeIo196SXYjFX>
  
  (You need a reddit account)
  We have used [Pushshift](https://pushshift.io/) to scrape the data from the subreddit. Unfortunately, it does not allow to extract their comments. To extract this information we use[PRAW](https://praw.readthedocs.io/en/latest/tutorials/comments.html) for this task.
  
  The featres we will use for this are-
  Feature Name | Description
  -------------|-------------
  id | id of the post
  title | title of the post
  url | url associated with the post
  author | author name
  score | score of the post (upvotes-downvotes)
  created_utc | timestamp of post
  num_comments | number of comments in the post
  permalink | permalink associated with the post
  link_flair_text | flair of the post
  over_18 | whether post is age restricted or not
  selftext | description of the post
  comments | comments of the post (LIMIT 10)
  
  Among ~41k samples from the scraped data, after applying preprocessing steps to the dataset we are left with ~30k samples.
  Splitting into train and test (70% and 30%)
  Description |Samples
  --------|--------
  Train | ~24000
  Test | ~6000
  
  We are considering 8 flairs-
  Label|Flair | Samples
  ---|---|---------
  1.|Politics|8425
  2.|Non-Political|6988
  3.|Coronavirus|6228
  4.|AskIndia|4328
  5.|Policy/Economy|1270
  6.|Science/Technology | 1202
  7.|Business/Finance | 1080
  8.|CAA-NRC | 1070
  
## Flair Classification
  
  ### Approach:<br/>
   1. Data is collected. <br/>
   2. The __title__, __comments__, __selftext__, __permalink__ are cleaned by removing bad symbols and stopwords using nltk. <br/>
   3.Splitting of data. <br/>
   4. We have considered five types of features as input- <br/>
   
          a) Title<br/>
          b) > Comments<br/>
          c) Title and Selftext<br/>
          d) Title, Comments and Selftext<br/>
          e) Title, Comments, Selftext and Permalink<br/>
          
   5. The following algorithms are applied on the dataset- <br/>
          
          a) Naive-Bayes Classifier
          b) Random Forest
          c) Linear SVM
          d) MLP Classifier
          e) Logistic REgression
          f) Bert Classifier
    
   6. Training and testing on the dataset showed __Bert Classifier__ has the best testing accuracy __60.17%__ when trained on the combination of __Title + Comments + Selftext + Permalink s feature.
   
   7. The best model is saved and is deployed in the Web App for the prediction of the flair from the URL of the post.
   
   ## RESULTS
   
   FEATURE | NB CLASSIFIER | LINEAR SVM | LOGISTIC REGRESSION |RANDOM FOREST | MLP CLASSIFIER
   -------|-------------|----------|-------------------|-------------|---------------
   __Title__|50.36|52.73|43.06|__54.74__|46.39
   __Comments__|32.42|__37.75__|34.27|36.91|32.84
   __Title + Selftext__|52.45|54.76|43.97|__55.36__|46.95
   __Title + Selftext + Comments__|48.99|54.43|45.92|__55.31__|46.13
   __Title + Selftext + Comments + Permalink__|49.32|55.12|50.20|__55.36__|48.30
   
   FEATURE | BERT
   ---------|----
   __Title__|58.32
   __Title + Selftext + Comments + Permalink__ |__60.17__
   
   ### Intuition behind combining feature
