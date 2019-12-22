from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd
import requests
from bs4 import BeautifulSoup

print('enter the stock to analyze')
s = input()
url = "http://www.google.com/search?q="+s+"&tbm=nws"
r1 = requests.get(url)
coverpage = r1.content
# print(coverpage)
soup1 = BeautifulSoup(coverpage, 'html5lib')
headlines = soup1.find_all('div', class_='BNeawe vvjwJb AP7Wnd')

sia = SIA()


##################### this part is taken from https://github.com/jasonyip184/StockSentimentTrading #########

####################modifies vader lexicon for financial stock persepective#####################

import csv
import pandas as pd

# stock market lexicon
stock_lex = pd.read_csv('lexicon_data/stock_lex.csv')
stock_lex['sentiment'] = (stock_lex['Aff_Score'] + stock_lex['Neg_Score'])/2
stock_lex = dict(zip(stock_lex.Item, stock_lex.sentiment))
stock_lex = {k:v for k,v in stock_lex.items() if len(k.split(' '))==1}
stock_lex_scaled = {}
for k, v in stock_lex.items():
    if v > 0:
        stock_lex_scaled[k] = v / max(stock_lex.values()) * 4
    else:
        stock_lex_scaled[k] = v / min(stock_lex.values()) * -4

# Loughran and McDonald
positive = []
with open('lexicon_data/lm_positive.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        positive.append(row[0].strip())

negative = []
with open('lexicon_data/lm_negative.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        entry = row[0].strip().split(" ")
        if len(entry) > 1:
            negative.extend(entry)
        else:
            negative.append(entry[0])

final_lex = {}
final_lex.update({word:2.0 for word in positive})
final_lex.update({word:-2.0 for word in negative})
final_lex.update(stock_lex_scaled)
final_lex.update(sia.lexicon)
sia.lexicon = final_lex

####################


results = []

for line in headlines:
    casted_line = str(line)[34:-6]
    pol_score = sia.polarity_scores(casted_line)
    pol_score['headline'] = casted_line
    results.append(pol_score)

results = pd.DataFrame(results)

print(results)
