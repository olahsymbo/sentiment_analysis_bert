# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:59:22 2019

@author: olahs
"""
  
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt 
#nltk.download()
from nltk.corpus import stopwords  
from wordcloud import WordCloud,STOPWORDS  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import RandomForestClassifier  
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


##### Get the data sets of from Sentiment.csv containing Positive, Negative, and Neutral Tweets #############
df = pd.read_csv("C:/Sentiment/SentimentData.csv")

data = df[['text','sentiment']]  ## get only the sentiment and text columns


####### Get the data for positive, negative, neutral jsut for plotting word cloud
data_pos = data[ data['sentiment'] == 'Positive']  # find data that have positive tweets
data_pos = data_pos['text']
data_neg = data[ data['sentiment'] == 'Negative'] # find data that have negative tweets
data_neg = data_neg['text']
data_neut = data[ data['sentiment'] == 'Neutral'] # find data that have neutral tweets
data_neut = data_neut['text']


### define the function for removing @ # RT from tweets
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@') 
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    #### initialize word cloud by using the clean words 
    wordcloud = WordCloud(stopwords=STOPWORDS,             
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(5, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

#### View the word cloud for pos, neg, neut tweets
print("Positive words")
wordcloud_draw(data_pos,'white')
print("Negative words")
wordcloud_draw(data_neg)
print("Neutral words")
wordcloud_draw(data_neut, 'blue')


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

X_trainn, X_testt, y_train, y_test = train_test_split(features, target.astype(int), random_state=42, test_size=0.3)


#########################################################
#########################################################
###### Prediction using Naive Bayes Classifier ##########
#########################################################
#########################################################
 
gnb = BernoulliNB()
gnb.fit(X_trainn, y_train)

y_pred = gnb.predict(X_testt)  

print("Test Accuracy of Naive Bayes Classifier :: ", accuracy_score(y_test, y_pred)) 



#########################################################
#########################################################
###### Prediction using Random Forest Classifier ########
#########################################################
#########################################################
 ## fit random forest classifier
clfr = RandomForestClassifier()
clfr.fit(X_trainn, y_train)

## predict random forest classifier
predictions = clfr.predict(X_testt)

print("Test Accuracy of Random Forest Classifier :: ", accuracy_score(y_test, predictions))
 

############################################################################
############## Now Let's Perform Independent Test of some Tweets ###########
##############                                           ###################
############## Prediction examples for a Positive Tweet ####################
print("Positive (1) Tweet Prediction")

Y = tfidf.transform(["thanks for the recent follow. Much appreciated happy  Get it."])
prediction = clfr.predict(Y)
print(prediction)


############## Prediction examples for a Negative Tweet  ####################
print("Negative (-1) Tweet Prediction")

Yn = tfidf.transform(["My area's not on the list unhappy  think I'll go LibDems anyway."])
prediction = clfr.predict(Yn)
print(prediction)


############## Prediction examples for a Neutral Tweet  ####################
print("Neutral (0) Tweet Prediction")

Yb = tfidf.transform(["Haryana peasants demand justice for right to cattle trade."])
prediction = clfr.predict(Yb)
print(prediction)