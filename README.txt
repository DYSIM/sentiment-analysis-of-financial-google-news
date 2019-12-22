Sentiment Analysis of Financial Stock news from Google news.

- Scrapes headlines from Google news
- Performs sentiment analysis with NLTK vader
- Prints out results in Pandas DataFrame


./lexicon_data and code to update vader lexicon is from https://github.com/jasonyip184/StockSentimentTrading. This adds Loughran-McDonald Financial Sentiment Word Lists to vader lexicon to take on financial stock perspective.


Dependencies:
nltk
Pandas
BeautifulSoup



run: python sentiment_analysis_of_financial_google_news.py
