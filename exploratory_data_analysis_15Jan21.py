# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 21:24:00 2022

@author: Andrew
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cat_adoption_clean_dataset_15Jan21.csv')



# Are there an equal number of males and females? Slightly more males but almost 50-50 which is good!
#df['sex'].value_counts(sort=True).plot(kind='barh')

# Which breed is more represented
#df['breed'].value_counts(sort=True).plot(kind='barh')
# Most are labelled domestic short, medium or long or mixed... (>80%)
# Those that have specific breed listed: there are 5 ragdoll or 4 sphynx cat 

# Which rescue groups are helping with cats (top 20)
#df['rescue_grp'].value_counts(sort=True)[:20].plot(kind='barh')
# If we are interested, we can visit these websites directly to figure out if we want to support them
# Heart and Soul, Give Kitty a home and Ballarat animal shelter provide quite a few but they are quite spread out!

# What are the locations of the cats
#df['location'].value_counts(sort=True)[:20].plot(kind='barh')
# Quite a few are from ballarat

# What is the range of fees? Lets plot a distplot
# sns.distplot(df['fee'], kde=False) 
# plt.title('Adoption fee', fontsize=18)
# plt.xlabel('Fee (dollars)', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)


# What is the range of cat ages? Lets plot a distplot
# sns.distplot(df['age_in_months'], kde=False) 
# plt.title('Age range of cats', fontsize=18)
# plt.xlabel('Age', fontsize=16)
# plt.ylabel('Frequency', fontsize=16)


# Is there a correlation betwwen age and fee?
sns.scatterplot(data=df, x ='age_in_months', y='fee')
plt.title('Correlation bettween age and fee', fontsize=18)
plt.xlabel('Fee (dollars)', fontsize=16)
plt.ylabel('Age (in months)', fontsize=16)
print("Correlation score between age and fee:")
print(df['age_in_months'].corr(df['fee']))  #slight negative correlation but  not exacly great -0.34