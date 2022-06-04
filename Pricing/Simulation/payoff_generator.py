import pandas as pd
import numpy as np
from path_generator import GeometricBrownianMotion
"""payoff generator, based on path generator"""
def CallOption(K):
    return lambda x:np.fmax(x-K,0)
def PutOption(K):
    return lambda x:np.fmax(K-x,0)
class EuropeanOption:
    "The payoff of European option can be computed as discounted payoff at time T"
    def fit_path(self,path):
        self.path=path

    def compute_payoff(self,r,func):
        """
        param r: risk-free interest
        param func: path independent payoff function
        return: payoff at time 0
        """
        self.payoff = self.path.copy(deep=True).apply(func)
        return np.exp(-path.columns[-1]*r)*self.payoff[path.columns[-1]].mean()
class AmericanOption:
    """American style option price cannot be computed as discounted average payoff"""
    """One common simulation based algorithm is Least Square Monte Carlo (LSMC) by Longstaff and Schwartz (2001) """
    def fit_path(self,path):
        self.path=path
    def compute_payoff(self,r,func,deg=5):
        """
        param r: risk-free interest
        param func: path independent payoff function
        return: payoff at time 0
        """
        self.payoff = self.path.copy(deep=True).apply(func)
        self.rule_matrix = self.path.applymap(lambda x:0).copy(deep=True)
        t_end=self.payoff.columns[-1]
        self.rule_matrix.loc[:,t_end] = (self.payoff[t_end]>0).astype(int) # create decision at T
        for t in range(len(self.payoff.columns)-1,1,-1):
            t_prev = self.payoff.columns[t-1] # reversed order
            t_now = self.payoff.columns[t]
            price_t0 = self.path.loc[self.payoff[t_prev]>0, t_prev].copy(deep=True)
            payoff_t1 = np.exp(-r*(t_now-t_prev))*self.payoff.loc[price_t0.index,t_now]
            LSMC = np.polynomial.laguerre.lagfit(x=price_t0.values,y=payoff_t1,deg=deg) # fit the polynomial suggested by Longstaff and Schwartz, while in practice this is flexible
            """update rule_matrix"""
            self.rule_matrix.loc[price_t0.index,t_prev] = (
                    (self.payoff[t_prev].loc[price_t0.index]) > np.polynomial.laguerre.lagval(x=price_t0.values,c=LSMC)
            ).astype(int).values
        """set previous decison to 0 if early exercise"""
        for t in range(len(self.payoff.columns)-1):
            for t_next in self.payoff.columns[t+1:]:
                self.rule_matrix.loc[self.rule_matrix[self.payoff.columns[t]]==1,t_next]=0
        """compute payoff using decison matrix"""

        discount_factor = np.exp(-r*self.payoff.columns)
        return np.fmax((self.rule_matrix*self.payoff).sum().dot(discount_factor)/len(self.payoff),self.payoff.iloc[0,0])



