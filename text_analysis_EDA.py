# -*- coding: utf-8 -*-

# Here, we are going to use scattertext to uncover any interesting trends from the text
# See : # https://github.com/JasonKessler/scattertext

import scattertext as st
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse.linalg import svds


# Read in the clean dataset
df = pd.read_csv("cat_adoption_clean_dataset_15Jan21.csv", encoding='cp1252')  

# I would like to do some preprocessing first, remove stopwords, lemmatize etc...

def preprocess_words(text): 
    """
    Removes URLs in text, makes everything lowercase, substitutes apostrophe, removes dodgy characters and lemmatizes token

    Parameters
    ----------
    text : string
        Text for preprocessing

    Returns
    -------
    string
        Processed string for scattertext visualization

    """
    # Initiate the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Initiate tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Add a few new stop words
    en_stop.extend(['cat', 'pet', 'foster'])
    
    # Remove all urls
    i = re.sub(r"(www|http:|https:)+\S+", "", text)
    
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
    
    # Concatanate the processed text into a string 
    return " ".join(stopped_tokens)

# Apply the preprocessing function to each row of the 'personality' column 
df['procesed_text']= df['personality'].apply(preprocess_words)

#%%
# First, lets look at a scatterplot of the terms associated with male and female cats
df = df.assign(parse=lambda df: df.procesed_text.apply(st.whitespace_nlp_with_sentences))

corpus = st.CorpusFromParsedDocuments(df, category_col='sex', parsed_col='parse').build().get_unigram_corpus().compact(st.AssociationCompactor(2000))

html = st.produce_scattertext_explorer(
    corpus,
    category='Male', category_name='Male', not_category_name='Female',
    minimum_term_frequency=5, pmi_threshold_coefficient=2,
    width_in_pixels=1000, metadata=corpus.get_df()['names'],
    transform=st.Scalers.dense_rank
)
open('./scatter_text_cat_sex.html', 'wb').write(html.encode('utf-8'))
# Although not very strong, 'quiet' 'sweet' 'trust' appears to be terms associated with females
# For males, it's things like 'big', 'handsome', 'outdoor'
# If we were interested in more statistically significant terms, we could use 'keyness' which is a chi-square test to determine assocaition of terms with sex 

#%%
# Document-Based Scatterplots
# Here we use SVD (a dimensionality reduction technique) to group similar cat profiles based on their singula values
corpus = corpus.add_doc_names_as_metadata(corpus.get_df()['names'])

# Get tf-idf scores of corpus words
embeddings = TfidfTransformer().fit_transform(corpus.get_term_doc_mat())

# Apply sparse svd and add the first two singular values to the dataframe
u, s, vt = svds(embeddings, k=3, maxiter=20000, which='LM')
projection = pd.DataFrame({'term': corpus.get_metadata(), 'x': u.T[0], 'y': u.T[1]}).set_index('term')

# Set the scattertext to show a plot of the grouped cat names (based on their singular values)  
category = 'Male'
scores = (corpus.get_category_ids() == corpus.get_categories().index(category)).astype(int)
html = st.produce_pca_explorer(corpus,
                               category=category,
                               category_name='Male',
                               not_category_name='Female',
                               metadata=df['names'],
                               width_in_pixels=1000,
                               show_axes=False,
                               use_non_text_features=True,
                               use_full_doc=True,
                               projection=projection,
                               scores=scores,
                               show_top_terms=False)

open('./scatter_text_cat_sex_svd_grouping.html', 'wb').write(html.encode('utf-8'))
# We do some groupings (mainly due to the length of the text and descriptions from each company)

#%%
# One last idea I had was to try and analyze only the adjectives in the description of the cats (beautiful, handsome, cute etc...) and group them to reduce possible 'noise'

# First extract all adjectives
import spacy
nlp = spacy.load('en_core_web_sm')

def adj_extractor(doc):
    """ Extract all the adjectives from text"""
    doc = nlp(doc)
    adjectives = []
    for token in doc:
        if token.pos_ == 'ADJ':
            adjectives.append(token)
    return ' '.join([str(w) for w in adjectives])

# Apply the adj_extractor function to create a new column called 'adjectives'
df['adjectives'] = df['personality'].apply(adj_extractor)

# Perform SVD with adjectives
df = df.assign(parse=lambda df: df.adjectives.apply(st.whitespace_nlp_with_sentences))
corpus = st.CorpusFromParsedDocuments(df, category_col='sex', parsed_col='parse').build().get_unigram_corpus().compact(st.AssociationCompactor(2000))
corpus = corpus.add_doc_names_as_metadata(corpus.get_df()['names'])
embeddings = TfidfTransformer().fit_transform(corpus.get_term_doc_mat())
u, s, vt = svds(embeddings, k=5, maxiter=20000, which='LM')
projection = pd.DataFrame({'term': corpus.get_metadata(), 'x': u.T[0], 'y': u.T[1]}).set_index('term')


category = 'Male'
scores = (corpus.get_category_ids() == corpus.get_categories().index(category)).astype(int)
html = st.produce_pca_explorer(corpus,
                               category=category,
                               category_name='Male',
                               not_category_name='Female',
                               metadata=df['names'],
                               width_in_pixels=1000,
                               show_axes=False,
                               use_non_text_features=True,
                               use_full_doc=True,
                               projection=projection,
                               scores=scores,
                               show_top_terms=False)

open('./scatter_text_cat_sex_svd_grouping_adjectives.html', 'wb').write(html.encode('utf-8'))


