#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:19:27 2020

@author: o.arigbabu
"""

import matplotlib.pyplot as plt 
from wordcloud import WordCloud,STOPWORDS

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