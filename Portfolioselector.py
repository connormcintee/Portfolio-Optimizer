import numpy as np
import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
with open('data.csv', 'r') as file:
    data=pd.read_csv(file)
date=data["Date"]
data=data.drop(columns="Date",axis=1)
df_stocks=pd.DataFrame(data=data)
stocks=list(df_stocks.columns)
mu = expected_returns.mean_historical_return(df_stocks)
#Sample Variance of Portfolio
Sigma = risk_models.sample_cov(df_stocks)
##
print(mu)
print(Sigma)
ef = EfficientFrontier(mu, Sigma, weight_bounds=(0,1))
pfolio=ef.max_sharpe()
#May use add objective to ensure minimum zero weighting to individual stocks
#weights=ef.clean_weights()
print(pfolio)
mu=[]
Sigma=[]
cov=[]
def mark(w,x):
    var=0
    for i in range(len(x)):
        for j in range(len(x)):
            var+=w[i]*w[j]*Sigma[x[i]][x[j]]
    std=np.sqrt(var)
    ret=0
    for i in range(len(x)):
        ret+=mu[x[i]]*w[i]
    return (ret/std)


def Optimizer(Sigma,w,x):
    pass