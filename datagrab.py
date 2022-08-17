from datetime import datetime
import pandas as pd
import pandas_datareader as web
import requests
import bs4 as bs
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker=ticker[:-1]
        tickers.append(ticker)
    tickers.remove('DOW')
    return tickers
def get500():
    tickers = save_sp500_tickers()
    del tickers[64]
    del tickers[79]
    #tickers = ['BSX','AES','BRK-B','SEE','QQQ','SPY']
    thelen = len(tickers)
    price_data = []
    start='1970-06-20'
    end = '2021-06-20'
    years=(datetime.strptime(end,"%Y-%m-%d").date()-datetime.strptime(start,"%Y-%m-%d").date()).days/365
    for ticker in range(thelen):
        print(ticker)
        try:
            prices = web.DataReader(tickers[ticker], start=start,data_source='yahoo')
            price_data.append(prices[['Adj Close']])
        except: print(tickers[ticker])
    df_stocks = pd.concat(price_data, axis=1)
    df_stocks.columns=tickers
    with open('data.csv', 'w') as outfile:
        df_stocks.to_csv(outfile, index=True)