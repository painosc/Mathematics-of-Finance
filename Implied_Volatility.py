#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy.stats as st
V_market = 2    
K        = 120  
T        = 1   
r        = 0.05 
S_0      = 100
sigma_0  = 0.25 

def ImpliedVolatility(S_0,K,sigma,tau,r):
    error    = 1e10; 
    optPrice = lambda sigma: BS_Call_Option_Price(S_0,K,sigma,T,r)
    vega= lambda sigma: dV_dsigma(S_0,K,sigma,T,r)
    
    n = 1.0 
    while error>10e-10:
        g         = optPrice(sigma) - V_market
        g_prim    = vega(sigma)
        sigma_new = sigma - g / g_prim
    
        error=abs(g)
        sigma=sigma_new;
        
        print('iteration {0} with error = {1}'.format(n,error))
        
        n= n+1
    return sigma

def dV_dsigma(S_0,K,sigma,T,r):
    #parameters and value of Vega
    d2   = (np.log(S_0 / float(K)) + (r - 0.5 * np.power(sigma,2.0)) * T) / float(sigma * np.sqrt(T))
    value = K * np.exp(-r * T) * st.norm.pdf(d2) * np.sqrt(T)
    return value

def BS_Call_Option_Price(S_0,K,sigma,tau,r):
    d1    = (np.log(S_0 / float(K)) + (r + 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(T))
    d2    = d1 - sigma * np.sqrt(T)
    value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * T)
    return value

sigma_imp = ImpliedVolatility(S_0,K,sigma_0,T,r)
message = '''Implied volatility for CallPrice= {}, strike K={}, 
      maturity T= {}, interest rate r= {} and initial stock S_0={} 
      equals to sigma_imp = {:.7f}'''.format(V_market,K,T,r,S_0,sigma_imp)
            
print(message)

val = BS_Call_Option_Price(S_0,K,sigma_imp,T,r)
print('Option Price for implied volatility of {0} is equal to {1}'.format(sigma_imp, val))


# In[ ]:




