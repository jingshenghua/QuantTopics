import pandas as pd
import numpy as np

"""object oriented path generator, based on numpy and pandas"""

class GeometricBrownianMotion:
    """simulate simple geometric brownian motion path with drift term and volatility term to be constant"""
    """ dS_t = mu*S_t*dt + sigma*S_t*dW_t, W_t stands for standard Brownian motion"""
    """https://en.wikipedia.org/wiki/Geometric_Brownian_motion"""
    def fit(self,mu,sigma):
        self.mu=mu
        self.sigma=sigma
    def __condition_mean(self,S_pre,mu_pre,sigma_pre,time_step):
        """
        param S_pre: previous spot price
        param mu_pre: previous drift rate
        param sigma_pre: previous volatility
        param time_step: constant time step, e.g dt
        return: the condition mean given previous spot, drift, and volatility
        """
        return np.log(S_pre) + (mu_pre - 0.5*sigma_pre**2)*time_step
    def __condition_std(self,sigma_pre,time_step):
        """
        param sigma_pre: previous volatility
        param time_step: constant time step, e.g dt
        return: the condition std given volatility and time step
        """
        return np.sqrt(sigma_pre**2*time_step)

    def simulate(self,s0,time_step,T,N=10000,Random_State=42):
        """
        param s0: initial stock spot price
        param time_step: constant time step, e.g dt
        param T: time to maturity
        param N: number of samples to generate
        param Random_State: fixed random state for simulation
        return: stock prices follow geometric Brownian motion from time 0 to time T
        """
        time_steps = np.arange(0,T+time_step,time_step,dtype=float)
        rand = np.random.RandomState(Random_State)
        samples = np.empty((N,len(time_steps)))
        samples[:,0]=s0 # set initial value

        """The next stock price log-normal distributed given previous"""
        for i in range(1,len(time_steps)):
            samples[:,i] = np.exp(
                rand.normal(
                    self.__condition_mean(S_pre=samples[:,i-1],mu_pre=self.mu,sigma_pre=self.sigma,time_step=time_step),self.__condition_std(sigma_pre=self.sigma,time_step=time_step)
                )
            )
        return pd.DataFrame(samples,index=range(1,N+1),columns=time_steps)










