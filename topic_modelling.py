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
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim import models

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

# Add a few new stop words
en_stop.extend(['cat', 'pet'])

# Lets remove the urls and perform other preprocessing 
# list for tokenized documents in loop
texts = []

# "L"oop through document list
for i in personality:
    
    # Remove all urls
    i = re.sub(r"(www|http:|https:)+\S+", "", i)
    
    # Make everything lowercase
    raw = i.lower()
    
    # Replace apostrophe and '-' with underscores 
    raw = re.sub("['-]", '_', raw)
    
    # Tokenize document string
    tokens = tokenizer.tokenize(raw)
    
    # Remove any tokens that are less than 1 character in length
    # This removes things like 'm' or 'd' or 's' toekns which dont have any meaning
    tokens_2_chr = [i for i in tokens if len(i) > 1]


    # Lemmatize tokens (so all words are standardized)
    lemma_tokens = [lemmatizer.lemmatize(i) for i in tokens_2_chr]
    
    # Remove stop words from tokens
    stopped_tokens = [i for i in lemma_tokens if not i in en_stop]
    
    # Add tokens to list
    texts.append(stopped_tokens)


# Create a dictionary
id2word = Dictionary(texts)

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
# I've set num_topics as 4 based on coherence scores of topics see later section
lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=4, 
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Use pyDavis to visualize data
pyLDAvis.enable_notebook()
# Transforms the topic model distributions and related corpus data into the data structures needed for the visualization
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word) 
pyLDAvis.save_html(vis, 'LDAModel.html')
# We see some of the topics like shy, fur, pat, trust, sweet, affectionate which could indicate characterisitcs of cat and also adoption info
# Other topics include details on adoption and names of the adoption centres


#%%
# We can also try and use the tf-idf scores instead of raw word counts to get words that are more 'unique' for each cat
# See here for info how to do LDA with tfidf: https://www.kaggle.com/rahulvks/eda-topic-modelling-lda-nmf-nltk-1

# Create the tfidf vectorizer and create a corpus using the vectorizer
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


# Build LDA model- this time using the tf-idf corpus
lda_model = LdaModel(corpus=corpus_tfidf,
                   id2word=id2word,
                   num_topics=4, 
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)


# Use pyDavis to visualize data
pyLDAvis.enable_notebook()
# Transforms the topic model distributions and related corpus data into the data structures needed for the visualization
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word) 
pyLDAvis.save_html(vis, 'LDAModel_tfidf.html')
# Similar results were obtained here for TF-IDF so I am not sure if it's worth developing topic modelling further with TF-IDF (for this project)

#%%
# Coherence score can be used to determine the optimal number of topics to use 
# https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Call function here for raw vector counts
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=20, step=2)
# Show graph
limit=20; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.title('Coherence score vs number of topics (count_vectorizer)')
plt.show()

# Call function here for tfidf scores
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus_tfidf, texts=texts, start=2, limit=20, step=2)
# Show graph
limit=20; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.title('Coherence score vs number of topics (tfidf vectorizer)')
plt.show()

# In both tfidf and raw counts, the optimal number appears to be ~4 topics 
