import praw
import logging
import os
import numpy as np
import flask
import joblib
from flask import Flask, render_template, request, jsonify
import re
import pandas as pd
import requests
import json
import csv
import time
import datetime
import sys
import sklearn.linear_model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import unicodedata
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

reddit=praw.Reddit(client_id='oCmAKw3zJDthfw',
client_secret='UcvYE3JbhkQNwMoB1V_j1tMNiu0',
user_agent='Flair Detection',
password='mmaannuu')

#



app=Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')


def home():
    return flask.render_template('index.html')
    predict()


"""remove html tags from text"""
def strip_html_tags(text):
    
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


contractions = { 
"Can't":"cannot",
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have",
'/r/india/comments/':''
}

contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expand_contractions(s, contractions_dict=contractions):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def string_form(value):
    return str(value)


def stop_words(text):
    stop_words = set(stopwords.words('english'))
    # tokens of words  
    word_tokens = word_tokenize(text) 

    filtered_sentence = [] 

    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 

            
    return " ".join(filtered_sentence)


def text_preprocessing(text):
    
    
    text = strip_html_tags(text)
    text = expand_contractions(text)
    text = remove_whitespace(text)
    text=normalizeString(text)
    text = text.lower()
    text=stop_words(text)
    text=string_form(text)
    
    return text


@app.route('/predict',methods=['POST'])
def predict():

    url = request.form['link']

    submission=reddit.submission(url=url)
    
    title=submission.title
    selftext=submission.selftext
    id_=submission.id
    permalink=submission.permalink
    flair= submission.link_flair_text

    
	if(title=="nan" or title=="NaN" or title=="[removed]" or title=='[deleted]'):
        title=" "
    if(selftext=="nan" or selftext=="NaN" or selftext=="[removed]" or selftext=='[deleted]'):
        selftext=" "
    if(permalink=="nan" or permalink=="NaN" or permalink=="[removed]" or permalink=='[deleted]'):
        permalink=" "
    if(flair=="nan" or flair=="NaN" or flair=="[removed]" or flair=='[deleted]'):
        flair=" "

    title_p=text_preprocessing(title)
    selftext_p=text_preprocessing(selftext)
    permalink_p=text_preprocessing(permalink)
    

    X=title_p+selftext_p+permalink_p

    
    
    loaded_model = joblib.load('model.sav')
    
    prediction = loaded_model.predict(X.split(" "))

    output = prediction[0]

    return render_template('index.html', prediction='{}'.format(output), prediction1= '{}'.format(flair))

# for endpoint
def predict_text(url,new_model):


    submission=reddit.submission(url=url)

    title=submission.title
    selftext=submission.selftext
    id_=submission.id
    permalink=submission.permalink
    flair= submission.link_flair_text

    

    if(title=="nan" or title=="NaN" or title=="[removed]" or title=='[deleted]'):
        title=" "
    if(selftext=="nan" or selftext=="NaN" or selftext=="[removed]" or selftext=='[deleted]'):
        selftext=" "

    if(permalink=="nan" or permalink=="NaN" or permalink=="[removed]" or permalink=='[deleted]'):
        permalink=" "

    if(flair=="nan" or flair=="NaN" or flair=="[removed]" or flair=='[deleted]'):
        flair=" "

    title_p=text_preprocessing(title)
    selftext_p=text_preprocessing(selftext)
    permalink_p=text_preprocessing(permalink)
    

    X=title_p+selftext_p+permalink_p

    

    prediction = new_model.predict(X.split(" "))

    output = prediction[0]
    return format(output)


#endpoint
@app.route('/automated_testing',methods=['POST'])
def test():

    

    data = request.files['file']

    links=data.readlines()

    key=[]
    value=[]

    loaded_model = joblib.load('model.sav')

    for i in range(0,len(links)):
        a=links[i].decode("utf-8")
        val=predict_text(a,loaded_model)
        key.append(a)
        value.append(val)

    dic={"Key":key,"Value":value}
    print(dic)
    return jsonify(dic)


    


if __name__ == "__main__":
    
    app.run(debug=True)
    