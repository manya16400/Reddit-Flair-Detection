# Reddit-Flair-Detection
  A Reddit Flair Detector web application to detect flairs of India subreddit posts using Machine Learning and NLP.<br>
  The entire code has been developed using Python programming language, utilizing its powerful text processing and machine learning modules. The application has been developed using Flask web framework and hosted on Heroku web server. The app can be used here [Reddit Flair Detection](https://flair-detector-india.herokuapp.com/).

## Table of Contents-
1. [About](#About)
2. [Installation](#Installation)
3. [Data](#Data)
4. [Flair Classifier](#Flair-Classifier)
5. [Deploying a Web app](#Deploying-a-Web-app)
6. [References](#References)
      
## About
  This repository illustrates the process of scraping reddit posts from the subreddit [r/india](https://www.reddit.com/r/india), text preprocessing/cleaning of data, building a classifier to classify the posts into 8 different flairs and deploying the best model as a web application.

## Installation
  Along with Python 3, this project requires following libraries (a few others which are mentioned in later section)  :
  * sklearn
  * Pytorch
  * pandas
  * numpy
  * matplotlib
  * nltk
  * praw
  * bs4
  * Flask
  * gunicorn
  
## Data
  To download the dataset
  <https://drive.google.com/open?id=1V2DaOj97SjZMTQa_MeJIeIo196SXYjFX>
  
  (You need a reddit account) <br>
  I have used [Pushshift](https://pushshift.io/) to scrape the data from the subreddit. Unfortunately, it does not allow to extract the comments of the post. To extract this information I used [PRAW](https://praw.readthedocs.io/en/latest/tutorials/comments.html).
  
  The features we will use for this are-
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
  Splitting into train and test (80% and 20%)-
  Description |Samples
  --------|--------
  Train | ~24000
  Test | ~6000
  
  Out of 40 flairs, we are considering __8 flairs__ because rest are less significant as they have <1000 entries.
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
  
## Flair Classifier
  
  #### Approach:<br/>
  1. The __title__, __comments__, __selftext__, __permalink__ are cleaned by removing bad symbols and stopwords using nltk. <br/>
  2. We have considered four types of features as input- <br/>
   
          a) Title
          b) Comments
          c) Title and Selftext
          e) Title, Selftext and Permalink
          
  3. The following algorithms are applied on the dataset- <br/>
          
          a) Naive-Bayes Classifier
          b) Random Forest
          c) Linear SVM
          d) MLP Classifier
          e) Logistic REgression
          f) Bert Classifier
    
   #### Results :
   
   FEATURE | NB CLASSIFIER | LINEAR SVM | LOGISTIC REGRESSION |RANDOM FOREST | MLP CLASSIFIER
   -------|-------------|----------|-------------------|-------------|---------------
   __Title__|50.367707|52.733737|43.062591|__54.747507__|44.696845
   __Comments__|32.423598|__37.653211__|34.270305|36.917797|33.453178
   __Title + Selftext__|52.459552|54.763850|43.977774|__55.368524__|47.409707
   __Title + Selftext + Permalink__|51.903905|__54.894590__|50.351364|54.633109|46.690635
   
   FEATURE | BERT
   ---------|----
   __Title__|57.95
   __Title + Selftext + Permalink__ |__62.510214__
   
   As Comments as a feature gave less accuracy, we do not combine comments with other features.
   
   4. _Inferences_- Training and testing on the dataset showed __Bert Classifier__ has the best testing accuracy __62.51%__ when trained on the combination of __Title + Selftext + Permalink__ as feature.<br>
   The tests shows that the combined features Title + Selftext + Permalink shows the best accuracy while Comments shows the worst accuracy. Machine learning models tries to detect specific words to identify the sentiment, hence title as a feature performs better than comments due to the fact that the title consists of all the keywords (selftext and permalink further contributes as they consist
   the short description of the post).
   
## Deploying a Web app
  As per the above results, Bert came out to be the best model and Random Forest to be the second best. But due to large size of bert and random forest we can not deploy it to Heroku web service (due to its limited slug size). So we consider the next best model __Linear SVM with accuracy of ~55% considering Title + Selftext + Permalink as feature__ for deploying to Heroku.
    
   Web screenshot
   ![Web app](https://github.com/manya16400/Reddit-Flair-Detection/blob/master/Web%20screenshots/2.png)
   Web endpoint screenshot (/automated_testing)
     ![](https://github.com/manya16400/Reddit-Flair-Detection/blob/master/Web%20screenshots/5.png)
## References

   1. <https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4>
   2. <https://medium.com/the-andela-way/deploying-a-python-flask-app-to-heroku-41250bda27d0>
   3. <https://praw.readthedocs.io/en/latest/tutorials/comments.html>
   4. <https://medium.com/@RareLoot/using-pushshifts-api-to-extract-reddit-submissions-fb517b286563>
   5. <https://towardsdatascience.com/nlp-text-preprocessing-a-practical-guide-and-template-d80874676e79>
