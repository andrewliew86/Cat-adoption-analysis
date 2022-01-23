# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:38:20 2022

@author: Andrew
"""

# In the first instance, we need to get a list of urls containing details of each cat
from urllib.request import urlopen as uReq
import urllib.request
from bs4 import BeautifulSoup as soup
import pandas as pd
from tqdm import tqdm # gives you a progress bar as you download stuff in a for loop
import time
from pathlib import Path
import os
import re

#%%
# Create list to store data
url_list = []


# Scrape data from pages 1 - 11 (range1,12)
# Note I am only focusing on cats and those that are in Victoria only 
for i in tqdm(range(1,12)):
    # sleep is used to make sure that I dont spam the server too much
    time.sleep(2)
    try:
        my_url = f"https://www.petrescue.com.au/listings/search/cats?interstate=true&page={i}&per_page=72&state_id%5B%5D=2"
        req = urllib.request.Request(my_url,headers={'User-Agent': "Magic Browser"})
        con = uReq(req)
        page_html = con.read()
        con.close()
        # html parsing
        page_soup = soup(page_html, 'html.parser')
        containers = page_soup.find_all(class_="cards-listings-preview")
        for container in containers:
            try:
                # get url for each cat's profile
                link_container = container.find_all('a', class_="cards-listings-preview__content")
                url_list.append(link_container[0]['href'])
            except IndexError:
                print('None')
                url_list.append('NA')
    except :
        continue  # In the case where there is a HTTP error or something...    
        
# Save urls into a csv file 
url_dict = {'url': url_list}  
df = pd.DataFrame(url_dict)  
current_path =  Path(__file__).parent.absolute()
df.to_csv(os.path.join(current_path, 'cat_urls_14Jan21.csv')) 
        
#%%
# Read in your scraped data into a list of urls from the csv file
cats_urls_df = pd.read_csv(os.path.join('cat_urls_14Jan21.csv'))
cats_urls = cats_urls_df['url'].tolist()

#%%
def scrape_details(tag, append_list_name):
    """ Custom function for scraping cat webpage details. tag: .css selector adddress, append_list_name: name of list to append results to"""
    try:
        # Use css selector
        data = page_soup.select(tag)[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
        print(data)
        append_list_name.append(data)
    except IndexError:
        print('N/A')
        append_list_name.append('NA')


# lists of cat details 
names = []
tag_lines = []
breeds = []
personality = []
other = []
more_info = []

for url in tqdm(cats_urls):
# sleep is used to make sure that I dont spam the server too much
    time.sleep(2)
    try:
        req = urllib.request.Request(url,headers={'User-Agent': "Magic Browser"})
        con = uReq(req)
        page_html = con.read()
        con.close()
        # html parsing
        page_soup = soup(page_html, 'html.parser')
        # Scrape cats data using my custom function
        scrape_details(".pet-listing__content__name", names)
        scrape_details(".pet-listing__content__feature", tag_lines)
        scrape_details(".pet-listing__content__breed", breeds)
        scrape_details(".personality", personality)
        scrape_details(".listing-details",other)
        
        # more_info is a bit different (these data contain lots of semi-structured so needs different handling)
        try:
            data = page_soup.select(".listing-details__table")[0].text.strip().encode('ascii', 'ignore').decode("utf-8")
            # Remove multiple newlines
            data = re.sub(r"\n+", "| ", data)
            print(data)
            more_info.append(data)
        except IndexError:
            print('N/A')
            more_info.append('NA')
    except :
        continue  # In the case where there is a HTTP error or something...
        
        
# There are multiple info in each list element of the 'other' list so lets collapse them using list comprehension before we create csv
other = [o.replace("\n", "| ") for o in other]


# Save daata into a csv file 
cat_dict = {'names': names, 
            'tag_lines':tag_lines, 
            'breeds': breeds, 
            'personality':personality,
            'other':other,
            'more_info':more_info
            }  

df = pd.DataFrame(cat_dict)  
current_path =  Path(__file__).parent.absolute()
df.to_csv(os.path.join(current_path, 'cat_raw_details_15Jan21.csv')) 






