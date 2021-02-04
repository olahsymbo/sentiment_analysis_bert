import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
projec_dir = os.path.realpath(os.path.dirname(app_path))
script_dir = os.path.dirname(projec_dir)

sys.path.insert(0, script_dir)

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from bert_serving.client import BertClient
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import Classifiers
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv(os.path.join(projec_dir, "SentimentData.csv"))

data = df[['text', 'sentiment']]  ## get only the sentiment and text columns


data_pos = data[data['sentiment'] == 'Positive']  # find data that have positive tweets
data_pos = data_pos['text']
data_neg = data[data['sentiment'] == 'Negative']  # find data that have negative tweets
data_neg = data_neg['text']
data_neut = data[data['sentiment'] == 'Neutral']  # find data that have neutral tweets
data_neut = data_neut['text']

tweets = []
stopwords_set = set(stopwords.words("english"))  # get the stopword dictionary

for index, row in data.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if
                               not word in stopwords_set]  ### break each tweet into set of words to ensure they are clean
    tweets.append((words_without_stopwords, row.sentiment))


clean_tweets = []
for i in range(len(tweets)):
    n1 = tweets[i]
    s = ' '
    clean_tweets.append(s.join(n1[0]))

target = data['sentiment']

#### set target as 1 for pos, 0 for neutr, -1 for neg
target[target == 'Positive'] = 1
target[target == 'Negative'] = -1
target[target == 'Neutral'] = 0


X_trainn, X_testt, y_train, y_test = train_test_split(clean_tweets, target.astype(int),
                                                      random_state=42, test_size=0.3)

# -------------------------BERT Model -------------------------------------#
print("starting BERT")
bc = BertClient(ip="SERVER_IP_HERE")
# get the embedding
X_train_bert = bc.encode(X_trainn.tolist())
X_test_bert = bc.encode(X_testt.tolist())


# -------------------------DNN --------------------------------------------#
# create model
print("starting DNN")
epoch = 50
batch = 150

# alternative model
sentiment_model = Sequential()
sentiment_model.add(Dense(500, input_dim=X_train_bert.shape[1], activation='relu'))
sentiment_model.add(Dense(400, activation='relu'))
sentiment_model.add(Dense(200, activation='relu'))
sentiment_model.add(Dense(100, activation='relu'))
sentiment_model.add(Dense(5, activation='sigmoid'))

Y = pd.get_dummies(y_train).values
Yt = pd.get_dummies(y_test).values
# Compile model
sentiment_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
sentiment_model.fit(X_train_bert, Y, epochs=epoch, batch_size=batch)
# evaluate the model
scores = sentiment_model.evaluate(X_test_bert, Yt)
print("\n%s: %.2f%%" % (sentiment_model.metrics_names[1], scores[1]*100))


sentiment_model.save(open(os.path.join(projec_dir, 'Bert_Model.h5')))
