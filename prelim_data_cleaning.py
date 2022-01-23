# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 11:33:00 2022

@author: Andrew
"""
import os
import pandas as pd
from pathlib import Path
import re 

# Set directory containing script as working directory
current_path =  Path(__file__).parent.absolute()
os.chdir(current_path)

# Load in scraped data
df = pd.read_csv('cat_raw_details_15Jan21.csv')
# Initial data cleaning

# Get sex and breed into different columns from the breeds column 
df['sex'] = df['breeds'].map(lambda x: x.split()[0])   # Sex is usually first
df['breed'] = df['breeds'].map(lambda x: ' '. join(x.split()[1:]))  # then domestic, hair etc..

# Get rescue group into seperate column
df['rescue_grp'] = df['other'].apply(lambda x: "".join(re.findall(r'''Rescue group\| ([A-Za-z0-9 _.,!"'/$&()-]+)\|*''', x)))

# Get location into seperate column 
df['location'] = df['other'].apply(lambda x: "".join(re.findall(r'''Location\| ([A-Za-z0-9 _.,!"'/$&()-]+)\|*''', x)))

# Get coat into seperate column
df['coat'] = df['other'].apply(lambda x: "".join(re.findall(r'''Coat\| ([A-Za-z0-9 _.,!"'/$&()-]+)\|*''', x)))

# Get date
df['date'] = df['other'].apply(lambda x: "".join(re.findall(r'''Last updated\| ([A-Za-z0-9 _.,!"'/$&()-]+)\|*''', x)))

# Adoption fee 
df['fee'] = df['other'].apply(lambda x: "".join(re.findall(r'''Adoption fee\| \$([0-9-.]+)\|*''', x)))

# Age is slightly more complicated
# First, age can be in weeks, months or years so needs to be standardized
df['age'] = df['other'].apply(lambda x: "".join(re.findall(r'''Age\| ([A-Za-z0-9 _.,!"'/$&()-]+)\|*''', x)))
                                                
# There is a mixture of years, months and weeks, lets change them all to months
def standardize_age(age):
    '''Standardize ages of cats to months from years and weeks. Input data is in the format of "1 year 3 months 2 weeks" etc...  
    Input: column containing age of cat'''
    # get 'years' if present, otherwise assign zero as value
    y = re.findall(r"(\d+) year[s]?", age)
    years = y[0] if y else 0
    # get 'months' if present, otherwise assign zero as value
    m = re.findall(r"(\d+) month[s]?", age) 
    months = m[0] if m else 0
    # get 'weeks' if present, otherwise assign zero as value
    w = re.findall(r"(\d+) week[s]?", age) 
    weeks = w[0] if w else 0
    # Now convert to weeks (e.g there are 12 months per year, 0.23 months per week)
    return int(months) + 12*int(years) + 0.23*int(weeks)


# Apply my standardize_age function
df['age_in_months'] = df['age'].apply(standardize_age)    

# The 'desexed', 'vaccinated', 'wormed' and 'interstate adoption' categories are not very helpful in the 'more info' column is not helpful as most are the same 
# The 'preference' for kids of a specific age might be helpful!
df['preference'] = df['more_info'].apply(lambda x: "".join(re.findall(r'''I'd prefer a home that\| ([A-Za-z0-9 _.,!"'/$&()-|]+)''', x)))

# Remove all the columns that are no longer needed
df.drop(['breeds', 'age', 'other', 'more_info'], axis=1, inplace=True)

# Export the 'clean' database
df.to_csv('cat_adoption_clean_dataset_15Jan21.csv')

