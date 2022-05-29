import pandas as pd
import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
def binomial(n,k):
    if 0<=k<=n:
        a=b=1
        for i in range(1,min(k,n-k)+1):
            a*=n
            b*=i
            n-=1
        return a//b
    else:
        return 0
"""object oriented tree generator, based on numpy and pandas"""

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

    def create_price_tree(self,s0,time_step,T):
        """
        param r: risk-free interest
        param q: asset dividend if any
        param sigma: annualized volatility term
        return: the binomial tree of stock prices as a list of numpy array
        """
        time_steps = np.arange(0,T+time_step,time_step)
        self.u =u= np.exp(self.sigma*np.sqrt(time_step))
        self.d= d = 1/u
        self.time_step=time_step
        self.p = (np.exp( (self.r-self.q) * time_step)-d)/(u-d)
        self.T=T
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

    def create_payoff_tree(self,price_tree,func,style='European'):
        """
        param price_tree: the binomial tree of stock prices
        param func: payoff function, e.g for call option = max(S-k,0), put option = max(K-S,0)
        param style: specify American or European style
        return: the binomial tree of payoff as a list of numpy array
        """
        payoff_tree = copy.deepcopy(price_tree)
        if style=='European':
            for i in range(len(price_tree)):
                payoff_tree[i]=func(price_tree[i])
        elif style=='American':
            for i in range(len(price_tree)):
                payoff_tree[i]=func(price_tree[i])
            for i in range(len(price_tree)-1,0,-1):
                current_node = payoff_tree[i]
                backward_current_node = [np.exp(-(self.r-self.q) * self.time_step) * (current_node[j]*self.p + current_node[j+1]*(1-self.p)) for j in range(len(current_node)-1)]
                prev_node = payoff_tree[i-1]
                payoff_tree[i-1] = np.fmax(prev_node,backward_current_node)
        return payoff_tree

    def compute_payoff(self,payoff_tree,style='European'):
        """
        param payoff_tree: the binomial tree of stock prices
        param func: payoff function, e.g for call option = max(S-k,0), put option = max(K-S,0)
        param style: specify American or European style
        return: the price given by binomial tree pricing
        """
        if style=='European':
            binomial_coef = np.array([binomial(len(payoff_tree)-1,i)*(self.p**i)*(1-self.p)**(len(payoff_tree)-i-1) for i in reversed(range(len(payoff_tree)))])
            payoff = np.sum(np.exp(-(self.r-self.q)*self.T)*binomial_coef*payoff_tree[-1])
            return payoff
        elif style=='American':
            return payoff_tree[0][0]

    def plot_tree(self,tree):
        """Visualization based on networkx"""
        plt.figure(figsize=(20, 14))
        G = nx.Graph()
        for i in range(0, len(tree) - 1):
            for j in range(1, i + 2):
                if i < len(tree):
                    G.add_edge((i, j), (i + 1, j))
                    G.add_edge((i, j), (i + 1, j + 1))
        posG = {}
        for node in G.nodes():
            posG[node] = (node[0], len(tree) + 2 + node[0] - 2 * node[1])
        labels = {}
        for node, pay in zip(G.nodes, np.concatenate(tree)):
            labels[node] = np.round(pay, 3)
        return nx.draw(G, pos=posG, labels=labels, with_labels=True, node_size=0, font_size=24)














