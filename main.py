import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
tbills=0.06
tickers = ['QQQ', 'BSX', 'XOM']
thelen = len(tickers)
price_data = []
for ticker in range(thelen):
    prices = web.DataReader(tickers[ticker], start='2015-01-01', end = '2020-06-06', data_source='yahoo')
    price_data.append(prices.assign(ticker=ticker)[['Adj Close']])
    df_stocks = pd.concat(price_data, axis=1)
df_stocks.columns=tickers
mean = expected_returns.mean_historical_return(df_stocks)
covariance = risk_models.sample_cov(df_stocks)
cov=df_stocks.cov()
def mark(w,x):
    var=0
    for i in range(len(x)):
        for j in range(len(x)):
            var+=w[i]*w[j]*covariance[x[i]][x[j]]
    std=np.sqrt(var)
    ret=0
    for i in range(len(x)):
        ret+=mean[x[i]]*w[i]
    return (std,ret)
ret=[]
std=[]
weights=[]
for i in range(1000):
    k=np.random.rand(thelen)
    w=k/sum(k)
    weights.append(w)
    res = mark(w, tickers)
    ret.append(res[1])
    std.append(res[0])
plt.plot(std,ret, "ro")
ef = EfficientFrontier(mean, covariance, weight_bounds=(0,1))
sharpe_pfolio=ef.max_sharpe()
sharpe_pwt=ef.clean_weights()
print(sharpe_pwt)
wt = list(sharpe_pwt.values())
vab=mark(wt,tickers,)
ret1=vab[1]
std1=vab[0]
std2=[]
ret2=[]
weights2=[]
index=[]
for i in range(len(ret)):
    for j in range(i,len(ret)):
        if ret[i]>ret[j] and std[i]<std[j]:
            std2.append(std[j])
            ret2.append(ret[j])
            index.append(j)
            weights2.append(weights[j])
        elif ret[j]>ret[i] and std[j]<std[i]:
            std2.append(std[i])
            ret2.append(ret[i])
            index.append(i)
            weights2.append(weights[i])
std3=[i for i in std if i not in std2]
ret3=[i for i in ret if i not in ret2]
weights3=[]
for i in range(len(weights)):
    if i!=index[i]:
        weights3.append(weights[i])
    plt.plot(std, ret,'ro' ,std3,ret3,'go',
        [0,std1,std1*1.5],[tbills,ret1,(ret1-tbills)/2+ret1],'b-',
            std1,ret1,'bo')
plt.ylabel('Return')
plt.xlabel('Risk')
plt.show()
