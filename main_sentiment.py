# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:59:22 2019

@author: olahs.
"""
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
projec_dir = os.path.realpath(os.path.dirname(app_path))
script_dir = os.path.dirname(projec_dir)

sys.path.insert(0, script_dir)

import pandas as pd  
import wordcloud_draw
from nltk.corpus import stopwords  
from wordcloud import WordCloud,STOPWORDS  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
import Classifiers
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


##### Get the data sets of from Sentiment.csv containing Positive, Negative, and Neutral Tweets #############
df = pd.read_csv(os.path.join(projec_dir, "SentimentData.csv"))

data = df[['text','sentiment']]  ## get only the sentiment and text columns


####### Get the data for positive, negative, neutral jsut for plotting word cloud
data_pos = data[ data['sentiment'] == 'Positive']  # find data that have positive tweets
data_pos = data_pos['text']
data_neg = data[ data['sentiment'] == 'Negative'] # find data that have negative tweets
data_neg = data_neg['text']
data_neut = data[ data['sentiment'] == 'Neutral'] # find data that have neutral tweets
data_neut = data_neut['text']


wordcloud_draw.wordcloud_draw(data, color = 'black')
    
#### View the word cloud for pos, neg, neut tweets
print("Positive words")
wordcloud_draw.wordcloud_draw(data_pos,'white')
print("Negative words")
wordcloud_draw.wordcloud_draw(data_neg)
print("Neutral words")
wordcloud_draw.wordcloud_draw(data_neut, 'blue')


tweets = []
stopwords_set = set(stopwords.words("english"))  # get the stopword dictionary
 
### loop through every tweet and remove stopwords, again remove @ # RT if still present
for index, row in data.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set] ### break each tweet into set of words to ensure they are clean
    tweets.append((words_without_stopwords, row.sentiment))

#### Merge the words from each tweet to form independent cleaned tweet in each row
clean_tweets = []
for i in range(len(tweets)):
    n1=tweets[i]
    s = ' '
    clean_tweets.append(s.join(n1[0]))



target = data['sentiment'] 

#### set target as 1 for pos, 0 for neutr, -1 for neg
target[ target == 'Positive'] = 1
target[ target == 'Negative'] = -1
target[ target == 'Neutral'] = 0

Data = clean_tweets

#########################################################################
########### Use TfidfVectorizer to encode (extract features) the Tweet ##
#########################################################################
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
features = tfidf.fit_transform(Data) 
print(features.shape)

X_trainn, X_testt, y_train, y_test = train_test_split(features, target.astype(int), 
                                                      random_state=42, test_size=0.3)

 
##### Prediction using Naive Bayes Classifier ########## 
n_estimators = 20
classifiers = Classifiers.Classifiers(n_estimators)

naive_class = classifiers.naiveBayes(X_trainn, y_train)
naive_pred = classifiers.predictor(naive_class, X_testt)
accuracy = classifiers.evaluator(naive_pred, y_test)
print("Test Accuracy of Naive Bayes Classifier :: ", accuracy) 

random_class = classifiers.randomForest(X_trainn, y_train)
random_pred = classifiers.predictor(random_class, X_testt)
accuracy = classifiers.evaluator(random_pred, y_test)
print("Test Accuracy of Random Forest Classifier :: ", accuracy)   

############## Now Let's Perform Independent Test of some Tweets ########### 
############## Prediction examples for a Positive Tweet ####################
print("Positive (1) Tweet Prediction")

Y = tfidf.transform(["thanks for the recent follow. Much appreciated happy  Get it."])
prediction = classifiers.predictor(random_class, Y)
print(prediction)


############## Prediction examples for a Negative Tweet  ####################
print("Negative (-1) Tweet Prediction")

Yn = tfidf.transform(["My area's not on the list unhappy  think I'll go LibDems anyway."])
prediction = classifiers.predictor(random_class, Yn)
print(prediction)


############## Prediction examples for a Neutral Tweet  ####################
print("Neutral (0) Tweet Prediction")

Yb = tfidf.transform(["Haryana peasants demand justice for right to cattle trade."])
prediction = classifiers.predictor(random_class, Yb)
print(prediction)