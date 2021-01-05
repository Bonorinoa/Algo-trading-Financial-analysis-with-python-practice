# Web scraper that returns the top 5 movers from yahoo finance

import bs4
from bs4.element import TemplateString
import requests
import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt



url = 'https://finance.yahoo.com/most-active?offset=0&count=100'

page = requests.get(url)
topMovers = bs4.BeautifulSoup(page.text, 'lxml')


headers = []

for i in topMovers.find_all('th'):
    title = i.getText()
    headers.append(title)

df = pd.DataFrame(columns = headers)


for j in topMovers.find_all('tr'): 
    row_data = [k.getText() for k in j]
    length = len(df)
    df.loc[length] = row_data


topMoversDataFrame = df.drop(df.index[0])

## Add column with date information to check it is running every morning
topMoversDataFrame['Time'] = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

## Change '% Change' column values to float to be able to manipulate them
topMoversDataFrame['% Change'] = list(map(lambda x: x[:-1], topMoversDataFrame['% Change'].values))
topMoversDataFrame['% Change'] = [float(x) for x in topMoversDataFrame['% Change'].values]

## Delete 52-week range and PE Ratio columns
topMoversDataFrame.drop(['PE Ratio (TTM)','52 Week Range'], axis=1,inplace=True)

## Sort dataframe by percentage change values. We want the ones that moved the most
topMoversDataFrame.sort_values('% Change', ascending=False ,inplace=True)


topMoversDataFrame.to_csv('stockScraped.csv')


