#We implement the Black-Schole pricing for European call via monte carlo and analytical

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

np.random.seed(1)
def GeneratePathsGBM(No_Paths, No_Steps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [No_Paths, No_Steps])
    X = np.zeros([No_Paths, No_Steps+1])
    S = np.zeros([No_Paths, No_Steps+1])
    time = np.zeros(No_Steps+1)
    
    X[:,0]=np.log(S_0)
    dt = T / float(No_Steps)
    
    for i in range(0, No_Steps):     
        X[:,i+1] = X[:, i] + (r - 0.5* sigma **2) *dt + sigma * np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i]+ dt
    
    S = np.exp(X)
    paths = { "time" : time, "X" : X, "S" : S}
    return paths
                   
def mainCalculation():
    No_Paths = 1000
    No_Steps = 50
    T = 1
    r = 0.05
    sigma = 0.25
    S_0 = 100
    K = 100
    
    Paths = GeneratePathsGBM(No_Paths, No_Steps, T, r, sigma, S_0)
    timeGrid = Paths["time"]
    X = Paths["X"]
    S = Paths["S"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(X))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    
    payoff = np.maximum(S[:,-1] - K, 0)
    Option_Price = np.mean(payoff)*(np.exp(-r * T))
    print('Simulated Option Value =', Option_Price)
    BS_Call_Option_Price(S_0, K, sigma, T,r)
    print('bs_Call option value =', BS_Call_Option_Price(S_0,K,sigma, T,r))

    
def BS_Call_Option_Price(S_0,K,sigma, T,r):
    d1 = (np.log(S_0 / float(K)) + (r - 0.5 * np.power(sigma,2.0)) * T) / float(sigma * np.sqrt(T))
    d2 = d1 + sigma * np.sqrt(T)
    value = st.norm.cdf(d2) * S_0 - st.norm.cdf(d1) * K * np.exp(-r * T)
    return value
mainCalculation()


# In[ ]:




