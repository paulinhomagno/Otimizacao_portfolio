import numpy as np

def capm_calc(columns: list, betas: dict, weights: dict, risk_market, tax_free_risk):
    

    sum_betas = 0
    
    for i in columns:

        sum_betas += (betas[i] * weights[i])

    capm = tax_free_risk + sum_betas * (risk_market - tax_free_risk)
    
    return capm