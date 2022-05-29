import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
from tree_generator import BinomialTree
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)
# headings
month = datetime.now().month
title = "Binomial Tree Option Pricing"
st.title(title + "🌲🎄")
st.sidebar.title("Parameters")
# user inputs on sidebar
S = st.sidebar.number_input('Stock Price (S)', value=100.,)
K = st.sidebar.number_input('Exercise Price (K)', value=100.,)
T = st.sidebar.number_input('Time Periods (T)', value=2., max_value=15.)
dt =  st.sidebar.number_input('Time step (dt)', value=1., max_value=15.,step=0.01)
r = st.sidebar.number_input('Inter-period Interest Rate (r)', value=0.05,)
q = st.sidebar.number_input('Dividend Yield (q)', value=0.0,)
sigma = st.sidebar.number_input('stock annualized volatility (sigma)', value=0.1,min_value=0.)

tree = BinomialTree()
tree.fit(r,q,sigma)
price_tree = tree.create_price_tree(S,dt,T)
st.sidebar.write("Stock Upper Factor (u) ", round(tree.u, 3))
st.sidebar.write("Stock Down Factor (d) ", round(tree.d, 3))
st.sidebar.write("Risk Neutral Probability (p) ", round(tree.p, 3))
# back to main body
st.header("*Cox-Ross-Rubinstein (CRR) binomial tree*")
st.markdown("This visualisation aims to explore the dynamics of CRR binomial tree in option pricing. "
            "https://en.wikipedia.org/wiki/Binomial_options_pricing_model"
            )
st.subheader('Key:')
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    price = st.checkbox('price tree')
with c2:
    European_call = st.checkbox('European Call tree')
with c3:
    European_put = st.checkbox('European Put tree')
with c4:
    American_call = st.checkbox('American Call tree')
with c5:
    American_put = st.checkbox('American Put tree')
# plot stock tree
if price:
    st.pyplot(tree.plot_tree(price_tree))
if European_call:
    payoff = tree.create_payoff_tree(price_tree,lambda x:np.fmax(x-K,0))
    st.pyplot(tree.plot_tree(payoff))
    st.write("European Call price ", round(tree.compute_payoff(payoff), 3))
if European_put:
    payoff = tree.create_payoff_tree(price_tree,lambda x:np.fmax(K-x,0))
    st.pyplot(tree.plot_tree(payoff))
    st.write("European put price ", round(tree.compute_payoff(payoff), 3))
if American_call:
    payoff = tree.create_payoff_tree(price_tree,lambda x:np.fmax(x-K,0),style='American')
    st.pyplot(tree.plot_tree(payoff))
    st.write("American Call price ", round(tree.compute_payoff(payoff,style='American'), 3))
if American_put:
    payoff = tree.create_payoff_tree(price_tree,lambda x:np.fmax(K-x,0),style='American')
    st.pyplot(tree.plot_tree(payoff))
    st.write("American Put price ", round(tree.compute_payoff(payoff,style='American'), 3))

st.subheader("Disclaimer")
st.write("All information aims to provide for educational purposes only and does not constitute financial advice")
