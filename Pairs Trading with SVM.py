
#Warning- there are a numerous parameters being run in this code, code takes long time to execute

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from time import sleep

bbh =  ['MRNA', 'AMGN', 'IQV', 'ICLR', 'VRTX', 'GILD', 'REGN', 'ILMN', 'CRL', 'SGEN', 'ILMN', 
            'BIIB', 'TECH', 'BNTX', 'BGNE', 'EXAS', 'ALNY', 'NVAX', 'QGEN', 'GH', 'NTRA', 'BMRN', 
            'INCY', 'TXG', 'NTLA', 'CRSP', 'BBH']
other_etfs = ['SPY', 'IBB', 'XBI', 'ARKG', 'FBT', 'LABU', 'IDNA', 'PBE', 'GNOM', 'BIB', 'SBIO', 'BTEC']
symbols = bbh + other_etfs

data = pd.read_csv("biotech_stock_prices.csv", index_col = 0)


# transform prices into cumulative returns
data = (data.pct_change()+1).cumprod()  #cumulative returns
data = data.iloc[1:]  #removing first row with NaNs
data = data / data.iloc[0]  #normalizing to new first row

# input data
s = ['MRNA', 'AMGN', 'IQV', 'ICLR', 'VRTX', 'GILD', 'REGN', 'ILMN', 'CRL',
     'SGEN', 'ILMN.1', 'BIIB', 'TECH', 'BNTX', 'BGNE', 'EXAS', 'ALNY',
     'NVAX', 'QGEN', 'GH', 'NTRA', 'BMRN', 'INCY', 'TXG', 'NTLA', 'CRSP',
     'SPY', 'IBB', 'XBI', 'ARKG', 'FBT', 'LABU', 'IDNA', 'PBE',
     'GNOM', 'BIB', 'SBIO', 'BTEC']
# target asset
y = ['BBH']

#PCA scree plot

from sklearn.decomposition import PCA

Xtmp = data[s] # select data without the target asset
pca = PCA(n_components=10)
pca.fit(Xtmp)

n_comp = np.arange(1,11)
plt.plot(n_comp, pca.explained_variance_ratio_)

#Parameter Selection
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from itertools import product

def compute_votes(data, pca_comp, beta, lookback, Cs, gammas, epsilons):
    '''
    compute daily votes of the models with the given parameters
    '''
    # start with equal weights
    weights = np.ones(len(Cs)*len(gammas)*len(epsilons))
    weights = weights/sum(weights) # normalize so that weights sum to 1

    daily_votes = np.zeros(len(data.index))

    for t in range(lookback,len(data.index)-1):
        predictions = []
        for C,gamma,epsilon in product(Cs,gammas,epsilons):
            model = make_pipeline(StandardScaler(), PCA(n_components=pca_comp), 
                                  SVR(C=C, gamma=gamma, epsilon=epsilon))
            X_train = data[s].iloc[t-lookback:t+1].values
            y_train = data[y].iloc[t-lookback:t+1].values.flatten()
            model.fit(X_train,y_train)
            X_test = data[s].iloc[t].values.reshape(1,-1)
            yhat = model.predict(X_test)
            predictions.append(yhat)
        # log all votes
        votes = -np.sign(data[y].iloc[t].values.flatten() - np.array(predictions)) # if price>fair, go short
        final_vote = np.dot(weights,votes)
        daily_votes[t] = final_vote   

        # update weights based on true direction
        true_direction = np.sign((data[y].iloc[t+1] - data[y].iloc[t]).values.flatten()) 
        if final_vote!=true_direction:
            incorrect_votes_ind = np.where(votes!=true_direction)[0]
            weights[incorrect_votes_ind] = beta * weights[incorrect_votes_ind]
            weights = weights/sum(weights)

    return daily_votes
   
# SVR hyperparameters
Cs = set((0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000))
gammas = set((0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000))
epsilons = set((0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1))


# strategy parameters
betas = [0.1,0.3,0.5,0.7] # betas for downgrading weights
lookbacks = [20] # how many last trading days to include in model training
pca_comps = [1] # number of principal components to use

columns = ['Beta', 'Lookback', 'PCA components', 'Num wins', 'Num losses', 'Pct Win', 
           'Avg Win', 'Avg Loss', 'Total Return', 'APR', 'Sharpe', 'Correlation with traded asset']

results = pd.DataFrame(columns = columns)

for pca_comp,beta,lookback in product(pca_comps,betas,lookbacks):
    daily_votes = compute_votes(data, pca_comp=pca_comp, beta=beta, lookback=lookback, 
                                Cs=Cs, gammas=gammas, epsilons=epsilons)
    
    datatmp = data[['BBH']].iloc[lookback+50:].copy() # skip first 50 days
    datatmp['vote'] = daily_votes[lookback+50:]
    datatmp['vote'] = datatmp['vote'].shift()
    datatmp['BBH_returns'] = datatmp['BBH'].pct_change()
    datatmp['alg_returns'] = (np.sign(datatmp['vote']) * datatmp['BBH_returns'])
    datatmp['alg_cumret'] = np.cumprod(datatmp['alg_returns']+1)
    
    datatmp.dropna(inplace=True)
    
    num_wins = (np.sign(datatmp[['BBH_returns']].values) == np.sign(datatmp[['vote']].values)).sum()
    num_losses = (np.sign(datatmp[['BBH_returns']].values) != np.sign(datatmp[['vote']].values)).sum()
    pct_win = num_wins / (num_wins + num_losses)
    avg_win = abs(datatmp[np.sign(datatmp['BBH_returns']) == np.sign(datatmp['vote'])]['BBH_returns']).sum()/num_wins
    avg_loss = abs(datatmp[np.sign(datatmp['BBH_returns']) != np.sign(datatmp['vote'])]['BBH_returns']).sum()/num_losses
    total_return = (datatmp['alg_cumret'][-1] - datatmp['alg_cumret'][0]) / datatmp['alg_cumret'][0]
    apr = (1+total_return)**(252/len(datatmp.index)) - 1
    sharpe = np.sqrt(252)*datatmp['alg_returns'].mean() / datatmp['alg_returns'].std()
    corrcoef = np.corrcoef(datatmp['BBH_returns'], datatmp['alg_returns'])[0,1]
    
    results = results.append({'Beta':beta, 'Lookback':lookback, 'PCA components':pca_comp, 
                              'Num wins':num_wins, 'Num losses':num_losses, 'Pct Win':pct_win, 
                              'Avg Win':avg_win, 'Avg Loss':avg_loss, 'Total Return':total_return, 
                              'APR':apr, 'Sharpe':sharpe, 'Correlation with traded asset':corrcoef}, 
                              ignore_index=True)
    
#####################################

###Model: beta=0.5, pca_comp=1
daily_votes = compute_votes(data, pca_comp=1, beta=0.5, lookback=20, 
                            Cs=Cs, gammas=gammas, epsilons=epsilons)
    
datatmp = data[['BBH']].iloc[lookback+50:].copy() # skip first 50 days
datatmp['vote'] = daily_votes[lookback+50:]
datatmp['vote'] = datatmp['vote'].shift()
datatmp['BBH_returns'] = datatmp['BBH'].pct_change()
datatmp['alg_returns'] = (np.sign(datatmp['vote']) * datatmp['BBH_returns'])
datatmp['alg_cumret'] = np.cumprod(datatmp['alg_returns']+1)
datatmp.dropna(inplace=True)    
    
datatmp['SPY'] = data.loc[datatmp.index, ['SPY']]

plt.figure(figsize=(18,6))
plt.plot(datatmp[['SPY']]/datatmp[['SPY']].iloc[0], label='Market (SPY)')
plt.plot(datatmp[['alg_cumret']]/datatmp[['alg_cumret']].iloc[0], label='Algorithm')
plt.legend()    

### Accounting for fees
# create an indicator of positions change
# -1 when we change position from long to short or vice versa
datatmp['pos_change'] = np.sign(datatmp['vote'])*np.sign(datatmp['vote'].shift())
datatmp['pos_change'][datatmp['pos_change']!=-1]=0
# subtract 0.2% fee from the return when we change position
datatmp['alg_returns_tc'] = datatmp['alg_returns'] + 0.002 * datatmp['pos_change']
# calculate cumulative returns with transaction costs
datatmp['alg_cumret_tc'] = np.cumprod(datatmp['alg_returns_tc']+1)

plt.figure(figsize=(18,6))
plt.plot(datatmp[['BBH']]/datatmp[['BBH']].iloc[0], label='Target asset (BBH)')
plt.plot(datatmp[['alg_cumret_tc']]/datatmp[['alg_cumret_tc']].iloc[0], label='Algorithm with TC')
plt.legend()

#Calculating metrics
def calculate_metrics(cumret):
    '''
    calculate performance metrics from cumulative returns
    '''
    total_return = (cumret[-1] - cumret[0])/cumret[0]
    apr = (1+total_return)**(252/len(cumret)) - 1
    sharpe = np.sqrt(252) * np.nanmean(cumret.pct_change()) / np.nanstd(cumret.pct_change())
    
    # maxdd and maxddd
    highwatermark=np.zeros(cumret.shape)
    drawdown=np.zeros(cumret.shape)
    drawdownduration=np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=cumret[t]/highwatermark[t]-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
    maxDD=np.min(drawdown)
    maxDDD=np.max(drawdownduration)
    
    return total_return, apr, sharpe, maxDD, maxDDD

   
metrics = pd.DataFrame(columns=['Total Return', 'APR', 'Sharpe', 'MaxDrawdown', 'MaxDrawdownDuration'], 
                       index=['BBH', 'SPY', 'Algo', 'Algo with TC'])

metrics.loc['BBH',:] = calculate_metrics(datatmp['BBH'])
metrics.loc['SPY',:] = calculate_metrics(datatmp['SPY'])
metrics.loc['Algo',:] = calculate_metrics(datatmp['alg_cumret'])
metrics.loc['Algo with TC',:] = calculate_metrics(datatmp['alg_cumret_tc'])


#correlations
datatmp[['BBH', 'SPY', 'alg_cumret', 'alg_cumret_tc']].pct_change().corr()