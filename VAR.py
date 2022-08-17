import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
tickers = ['XOM','QQQ','BRK-B','UNH','SPY']
thelen = len(tickers)
price_data = []
start='2005-06-20'
end = '2013-06-20'
years=(datetime.strptime(end,"%Y-%m-%d").date()-datetime.strptime(start,"%Y-%m-%d").date()).days/365
for ticker in range(thelen):
   prices = web.DataReader(tickers[ticker], start=start, end = end, data_source='yahoo')
   price_data.append(prices[['Adj Close']])
df_stocks = pd.concat(price_data, axis=1)
df_stocks.columns=tickers
# Get Annualized Return from Historical Data
mu = expected_returns.mean_historical_return(df_stocks)
# Get Covariance Matrix
sig = risk_models.sample_cov(df_stocks)
# Max Sharpe Ratio: Tangent to the Efficiency Frontier
ef = EfficientFrontier(mu, sig, weight_bounds=(0, 1))
pfolio=ef.max_sharpe()
# Clean weights
weights=ef.clean_weights()
print(weights)
ticker_rx2 = []
wt = list(weights.values())
wt=np.array(wt)
portret=df_stocks.dot(wt)
df_stocks['Portfolio']=portret
newtick=[*tickers,"Portfolio"]
for a in range(thelen+1):
    ticker_rx = df_stocks[[newtick[a]]].pct_change()
    ticker_rx = (ticker_rx+1).cumprod()
    ticker_rx2.append(ticker_rx[[newtick[a]]])
ticker_final = pd.concat(ticker_rx2,axis=1)
plt.figure(0)
for i, col in enumerate(ticker_final.columns):
    if col=="Portfolio": ticker_final[col].plot(linewidth=2)
    else: ticker_final[col].plot(linewidth=0.1+wt[i], linestyle='--' if wt[i]>0 else ':')
plt.title('Cumulative Returns')
plt.xticks(rotation=80)
plt.legend(ticker_final.columns)
#plt.show()
#Taking Latest Values of Return
pret = []
pre1 = []
price =[]
for x in range(thelen):
    pret.append(ticker_final.iloc[[-1],[x]])
    price.append((df_stocks.iloc[[-1],[x]]))
pre1 = pd.concat(pret,axis=1)
pre1 = np.array(pre1)
price = pd.concat(price,axis=1)
varsigma = pre1.std()
rtn=pre1.dot(wt)
dstand= (np.dot(wt.T, np.dot(sig / 250, wt))) ** 0.5
ystand = (np.dot(wt.T, np.dot(sig, wt))) ** 0.5
stand = (np.dot(wt.T, np.dot(sig * years, wt))) ** 0.5
print('The weighted expected portfolio return for selected time period is ' + str(round(rtn[0] * 100, 2)) + '%')
yrtn = (rtn) ** (1 / years) - (1)
drtn = (rtn) ** (1 / (years * 250)) - (1)
print('The weighted expected Yearly portfolio return for selected time period is ' + str(round(yrtn[0] * 100, 2)) + '%')
price=price.dot(wt)
print('Return: ', round(yrtn[0],2), '  Std: ',round(ystand,2))
lt_price=[]
final_res=[]
finaldist=[]
n=1000
for i in range(n):
    days = []
    ret = []
    dret=1
    y=5
    for day in range(y*250):
        r=np.random.normal(rtn / (years * 250), dstand)
        dret=float(r)*dret+dret
        ret.append(dret)
        days.append(day)
        if day==y*250-1:
            finaldist.append(dret)
    plt.figure(1)
    plt.plot(days,ret)
    plt.figure(2)
    plt.plot(days, np.log(ret))
plt.figure(1)
plt.title(str(n)+" Simulations of return over "+str(y)+" years")
plt.xlabel("Trading days")
plt.ylabel("Return on initial investment")
plt.figure(2)
plt.title(str(n)+" Simulations of log(return) over "+str(y)+" years")
plt.xlabel("Trading days")
plt.ylabel("Return on initial investment")
plt.figure(3)
plt.title("Distribution of Returns at the end of "+str(y)+" years")
plt.hist(finaldist,bins=int(2*(n**0.5)))
plt.axvline(sum(finaldist)/len(finaldist), color='k', linestyle='dashed', linewidth=3)
plt.axvline(np.percentile(finaldist,5), color='k', linestyle='dashed', linewidth=1)
plt.axvline(np.percentile(finaldist,95), color='k', linestyle='dashed', linewidth=1)
print("With the given Portfolio, after "+str(y)+" years:\n")
print("The Mean return is: ",sum(finaldist)/len(finaldist))
print("The 5th Percentile is: ",np.percentile(finaldist,5))
print("The 95th Percentile is: ",np.percentile(finaldist,95))
plt.show()