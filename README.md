# Cat adoption text analysis and fee prediction

Background: There are lots of cats that are available for adoption and it can be difficult to choose one based on the profiles that are available online.
I wanted to see if we could summarize cat characteristics based on their text descriptions and also determine if we could predict adoption fees based on the characteristics of the cat. 
We scraped a website to obtain cat profiles, performed some EDA and visualization with pandas before exploring the text data (using scattertext, topic modelling) and used basic machine learning tools for analyzing adoption fees.

Results: Preliminary EDA showed that Craigiburn and Pakernham (Melbourne suburbs) seem to be standout places for adoptions while certain cat breeds (non domestic cat) can be quite expensive. Analyzing a scatter of terms associated with males and females (using scattertext), we find terms like 'quiet' 'sweet' 'trust' to be associated with females
while 'big', 'handsome' and 'outdoor' tended to be used for males (Figure 1). However, these associations were not very strong. Topic modelling (Figure 2) showed two broad topics related to cat characteristics and two possibly related to administrative text related to adoptions. 
We also loosely grouped certain cats based on their profile descriptions using SVD so that we could shortlist similar cat profiles that matched our criteria. Lastly, we attempted to predict adoption fees using different ML techniques. We found age was the biggest contributing factor to adoption fees but the lack of data made it difficult to accurately predict adoption fees.

Python libraries/tools: Standard Python libraries along with libraries for text analysis (Scattertext, Gensim, pyLDAvis) and basic ML (Scikit-learn, XGBoost)

Figure 1: Visualizing terms assocaited with male or female cats using scattertext
![alt text](https://github.com/andrewliew86/Cat-adoption-analysis/blob/main/scatter_text.PNG)

Figure 2: Topic modelling example results visualized with pyLDAvis
![alt text](https://github.com/andrewliew86/Cat-adoption-analysis/blob/main/LDA_data.PNG)
