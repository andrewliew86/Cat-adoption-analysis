# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 21:58:41 2022

@author: Andrew
"""
# Based on tutorial from: https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer 
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer

from pprint import pprint
import re 
import pyLDAvis
import pyLDAvis.gensim

import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv('cat_adoption_clean_dataset_15Jan21.csv')

# We want to do some topic modelling of the 'personality' column
# Create a list of all the personalities of the cat as a list 
personality = df.personality.values.tolist()

# Initiate the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Initiate tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')


# Lets remove the urls and perform other preprocessing 
# list for tokenized documents in loop
texts = []

# "L"oop through document list
for i in personality:
    
    # Remove all urls
    i = re.sub(r"(www|http:|https:)+\S+", "", i)
    
    # Make everything lowercase
    raw = i.lower()
    
    # Tokenize document string
    tokens = tokenizer.tokenize(raw)

    # Remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # Lemmatize tokens (so all words are standardized)
    lemma_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]
    
    # Add tokens to list
    texts.append(lemma_tokens)

# Create a dictionary
id2word = Dictionary(texts)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
# Note that we will neeed to optimize the num_topics 
lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=10, 
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


#import pyLDAvis.gensim  # Not sure why using pyLDAvis.gensim didnt work; needed to be imported explicitly. This is a known issue. 
pyLDAvis.enable_notebook()
# Transforms the topic model distributions and related corpus data into the data structures needed for the visualization
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word) 
pyLDAvis.save_html(vis, 'LDAModel.html')


#%%
# We can also try and use the tf-idf scores instead of raw word counts for better results

# Create a tfidf object and then use the object on my list of lemma
tfidf = TfidfVectorizer(tokenizer=lambda x:x, stop_words='english', lowercase=True)    
tfidf.fit_transform(texts)
# See here for info how to do LDA with tfidf: https://www.kaggle.com/rahulvks/eda-topic-modelling-lda-nmf-nltk-1


