
import pandas as pd
import numpy as np
from scipy import stats

def monte_carlo_projection(data_returns: pd.DataFrame, amount):
    
    # returns percentage
    returns_pct = np.log1p(data_returns[['sharpe otm','cla otm','max_sqrt otm','hrp otm']].pct_change())
    returns_pct.fillna(0, inplace=True)
    
    # calculcates statistics
    mean_mc = returns_pct.mean()
    var_mc = returns_pct.var()
    drift = mean_mc - (0.5 * var_mc)
    std_mc = returns_pct.std()
    
    
    # simulation params 
    years = 3
    days = 252 * years
    simulation = 1000
    
    # random volatility by multidimensional normal distribution
    np.random.seed(0)
    Z = stats.norm.ppf(np.random.rand(days, simulation))
    daily_sharpe = np.exp(drift[0] + std_mc[0] * Z)
    
    np.random.seed(1)
    Z = stats.norm.ppf(np.random.rand(days, simulation))
    daily_risk = np.exp(drift[1] + std_mc[1] * Z)
    
    np.random.seed(2)
    Z = stats.norm.ppf(np.random.rand(days, simulation))
    daily_mqu = np.exp(drift[2] + std_mc[2] * Z)
    
    np.random.seed(3)
    Z = stats.norm.ppf(np.random.rand(days, simulation))
    daily_hrp = np.exp(drift[3] + std_mc[3] * Z)
    
    # the last row every column to predict
    pred_sharpe = np.zeros_like(daily_sharpe)
    pred_sharpe[0] = amount #data_returns['sharpe otm'][-1]

    pred_risk = np.zeros_like(daily_risk)
    pred_risk[0] = amount #data_returns['cla otm'][-1]

    pred_mqu = np.zeros_like(daily_mqu)
    pred_mqu[0] = amount #data_returns['max_sqrt otm'][-1]

    pred_hrp = np.zeros_like(daily_hrp)
    pred_hrp[0] = amount #data_returns['hrp otm'][-1]
    
    # calculate projections every day
    for day in range(1, days):
      
        pred_sharpe[day] = pred_sharpe[day - 1] * daily_sharpe[day]
        pred_risk[day] = pred_risk[day - 1] * daily_risk[day]
        pred_mqu[day] = pred_mqu[day - 1] * daily_mqu[day]
        pred_hrp[day] = pred_hrp[day - 1] * daily_hrp[day]
        
    return [pred_sharpe, pred_risk, pred_mqu, pred_hrp]