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
print(df['age_in_months'].corr(df['fee']))  
# Slight negative correlation so older cats tend to be cheaper (but not exacly all that strong -0.34)


# Lets look at sex, breed, rsc_grp and location to see if there are any effects on adoption fees
# There are too many categories for breed, rsc_grp and location so I am going to only look at the top 10 and then the rest I am going to put into 'other'
# This is mainly for visusalization purposes but might also help with machine learning later
need_breed = df['breed'].value_counts().index[:11]
df['breed_rdc_category'] = np.where(df['breed'].isin(need_breed), df['breed'], 'OTHER')

need_rsc_grp = df['rescue_grp'].value_counts().index[:11]
df['rsc_grp_rdc_category'] = np.where(df['rescue_grp'].isin(need_rsc_grp), df['rescue_grp'], 'OTHER')

need_loc = df['location'].value_counts().index[:11]
df['location_rdc_category'] = np.where(df['location'].isin(need_loc), df['location'], 'OTHER')

sns.catplot(x="fee", y="breed_rdc_category", data=df, kind='box', orient="h")
# Some of the breeds e.g. Sphynx, Egyptian Mau cat and ragdoll are very expensive but most are around 200

sns.catplot(x="fee", y="rsc_grp_rdc_category", data=df, kind='box', orient="h")
# Strong hears farm sanctuary , Give Kitty a Home, Heart & Soul and Maneki Neko Cat Rescue are generally pricier than other rescue groups

sns.catplot(x="fee", y="location_rdc_category", data=df, kind='box', orient="h")
# Craigiburn and Pakernham seem to be standout places for adoptions

#%% 
# Lets try using random_forest regressor to see if we can estimate fees
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# First, lets remove any rows with NaNs in fees
df.dropna(subset=['fee'], inplace=True)

# Make the training and test datasets
X = df[['sex', 'breed_rdc_category', 'location_rdc_category', 'rsc_grp_rdc_category','age_in_months']]
y = df['fee'].values

# Now, lets deal with the categorical data
# Get a list of columns that are 'objects' which would then be transformed through pandas dummy encoding
features_to_encode = X.columns[X.dtypes==object].tolist()

# Convert the features via get_dummies
def encode_and_bind(original_dataframe, features_to_encode):
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([dummies, original_dataframe], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return(res)

# Get a new transformed dataframe called X_encoded
X_encoded = encode_and_bind(X, features_to_encode)


# Instantiate a random forest regressor
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)

# Create your train and test sets 
X_train, X_test, y_train, y_test =  train_test_split(X_encoded, y, test_size=0.2)

# Fit rf to your training data
rf.fit(X_train, y_train)
            

# Evaluate the test set mean squared error using 3-fold cross validation
test_scores = cross_val_score(rf, X_test, y_test, scoring='neg_root_mean_squared_error', cv=3)

# Print scores
print("Using 3-fold cross-val, RMSE of %0.2f with a standard deviation of %0.2f " % (test_scores.mean()*-1, test_scores.std()))
# We get an RMSE of 99.14 with a standard deviation of 23.40 using 3-fold cross validation

# One of the great things about random forest is that it can tell you which features are most important
feature_importances = list(zip(X_train, rf.feature_importances_))
# Sort the feature importances by most important first
feature_importances_ranked = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Plot the top 25 feature importance
feature_names_25 = [i[0] for i in feature_importances_ranked[:25]]
y_ticks = np.arange(0, len(feature_names_25))
x_axis = [i[1] for i in feature_importances_ranked[:25]]
plt.figure(figsize = (10, 14))
plt.barh(feature_names_25, x_axis)   #horizontal barplot
plt.title('Random Forest Feature Importance (Top 25)',
          fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
plt.xlabel('Features',fontdict= {'fontsize' : 16})
plt.show()
# As suspected, age is the biggest predictor for fees

#%%
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# First, we deal with the categorical data. So I am using label encoder for this one (instead of one hot encoding)
encode_df = X.apply(LabelEncoder().fit_transform)

# You need to convert data into a specialized/optimized data structure called DMatrix  
data_dmatrix = xgb.DMatrix(data=encode_df,label=y, enable_categorical=True)


params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=70,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

# The best I can get is ~104 for the test set so I think this is probably the best we can get without hyperparameter tuning

# Lets visualize boosting trees and feature importances  
#xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=70)
#xgb.plot_tree(xg_reg,num_trees=0)
#plt.show()
