import pandas as pd
import numpy as np

"""object oriented path generator, based on numpy and pandas"""

class BinomialTree:
    """simple binomial tree with drift term and volatility term to be constant"""
    """                    S*u   
     one step example: S ->
                           S*d
    """
    """https://en.wikipedia.org/wiki/Binomial_options_pricing_model"""
    def fit(self,r,q,sigma):
        """
        param r: risk-free interest
        param q: asset dividend if any
        param sigma: annualized volatility term
        """
        self.r=r
        self.q=q
        self.sigma=sigma

    def create_tree(self,s0,time_step,T):
        """
        param r: risk-free interest
        param q: asset dividend if any
        param sigma: annualized volatility term
        return: the binomial tree as a list of numpy array
        """
        time_steps = np.arange(0,T+time_step,time_step)
        self.u =u= np.exp(self.sigma*np.sqrt(time_step))
        self.d= d = 1/u
        """initialize tree as a list of numpy array"""
        tree = [np.empty(1) for i in range(len(time_steps))]
        tree[0][0]=s0
        for i in range(1,len(tree)):
            tree[i]=s0*u**(np.arange(i,-1,-1))*d**np.arange(0,i+1,1)
        self.tree=tree
        return tree

    def print_tree(self):
        for node in self.tree:
            print(node)



