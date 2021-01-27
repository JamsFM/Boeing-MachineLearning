import nltk  
import numpy as np  
import random  
import string

#import bs4 as bs  
#import urllib.request  
import re

# Open file of log messages
log = open('logs.txt', 'r')

# Splits file into individual messages by newline
messages = log.read().split('\n')

# Removes empty space/lines at the end of the log file
while messages [len(messages)-1] == '':
    del messages [len(messages)-1]
#print(messages)
#print(len(messages))

# Removes punctuation and extra spaces
for i in range(len(messages)):
    messages [i] = messages [i].lower()
    messages [i] = re.sub(r'\W',' ',messages [i])
    messages [i] = re.sub(r'\s+',' ',messages [i])

print(messages)
#print(len(messages))

# Tokenizes log messages and counts the frequency of each word
wordfreq = {}
for sentence in messages:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

#print(wordfreq)