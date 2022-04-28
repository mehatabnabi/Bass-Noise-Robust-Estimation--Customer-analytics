#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# read data
df = pd.read_excel('adoptionseries2_with_noise.xlsx')
#df.head()

#Begin Function 1#

def extrapolate_discrete_bass(df, p, q, M, t):
    
### function to extrapolate a dataframe using discrete bass model to some time, t ###   

    for i in range(1, len(df)):
        df.loc[i, 'R(t)'] = np.nan
        df.loc[i, 'F(t)'] = np.nan
        
    for i in range(len(df), 50):  #extrapolation
        df.loc[i, 't'] = i+1
        df.loc[i, 'A(t)'] = df.loc[i-1, 'N(t)'] + df.loc[i-1, 'A(t)']
        df.loc[i, 'A(t)^2'] = df.loc[i-1, 'A(t)'] ** 2
        df.loc[i, 'R(t)'] = M - df.loc[i, 'A(t)']
        df.loc[i, 'F(t)'] = p + q * (df.loc[i, 'A(t)'] / M)
        df.loc[i, 'N(t)'] = df.loc[i, 'F(t)'] * df.loc[i, 'R(t)']
 
    return df # return extrapolated dataframe

# End function 1#

#Begin Function 2#

def extrapolate_cont_bass(df, p, q, M, t):
    
    ### function to extrapolate a dataframe using continuous bass model to some time, t ###

    for i in range(1, len(df)):
        df.loc[i, 'R(t)'] = np.nan
        df.loc[i, 'F(t)'] = np.nan

    for i in range(14, 50):
        t = i+1
        df.loc[i, 't'] = t
        A_t_num = 1 - np.exp((-1)*(p+q)*t)
        A_t_denom = 1 + (q/p)*np.exp((-1)*(p+q)*t)
        A_t = M * (A_t_num/A_t_denom)
        df.loc[i, 'A(t)'] = A_t
        df.loc[i, 'A(t)^2'] = A_t ** 2
        df.loc[i, 'R(t)'] = M - A_t
        df.loc[i, 'F(t)'] = p + q * (A_t/M)
        df.loc[i, 'N(t)'] = df.loc[i, 'F(t)'] * df.loc[i, 'R(t)']
        
    return df # return extrapolated dataframe

#End Function 2#

print('QUESTION 1.1')
df.loc[0, 'A(t)'] = 0
for i in range(1, len(df)):
	df.loc[i, 'A(t)'] = df.loc[i-1, 'N(t)'] + df.loc[i-1, 'A(t)']

df['A(t)^2'] = df['A(t)'] ** 2

# store intermediate df for question 1.2
df_2 = df.copy()
# store intermediate df for question 2
df_q2 = df.copy()

# initializing regression model
lin_reg = LinearRegression()
X = df.drop(columns=['t', 'N(t)'])
# features = X.columns
Y = df['N(t)']
# fit model
lin_reg.fit(X, Y)

# assigning the parameter values
a = lin_reg.intercept_
b, c = lin_reg.coef_[0], lin_reg.coef_[1]
p = (np.sqrt(b**2 - 4*a*c) - b) / 2
q = (np.sqrt(b**2 - 4*a*c) + b) / 2
M = -q / c

# extrapolation
df = extrapolate_discrete_bass(df, p, q, M, 30)
# display values
print(f'p = {round(p,3)}')
print(f'q = {round(q,3)}')
print(f'M = {round(M,3)}')
print(f'N(30) = {round(df.loc[29, "N(t)"],3)}')
print()

print('QUESTION 1.2')
# initializing the function to pass into curve_fit()
def compute_N(A_t, p, q):
    # return N(t) as a function of A_t #
    return 100*p + (q-p)*A_t - (q/100)*(A_t**2)

# define x and y
x = df_2['A(t)']
y = df_2['N(t)']
# run nonlinear regression
p_opt, p_cov = curve_fit(compute_N, x, y, p0 = [0.02, 0.5])
# extract p and q from optimal parameters
p, q = p_opt[0], p_opt[1]
# display values
print(f'M = {round(100,3)}')
print(f'p = {round(p,3)}')
print(f'q = {round(q,3)}')
print()

print('QUESTION 1.3')
# reinitializing M
M = 100
# extrapolation
df_2 = extrapolate_discrete_bass(df_2, p, q, M, 30)

# display value
print(f'N(30) = {round(df_2.loc[29, "N(t)"],3)}')
print()

print('QUESTION 1.4')
# start from initial data since we now have a new way of calculating A(t)
df_4 = pd.read_excel('adoptionseries2_with_noise.xlsx')

def compute_N(t, p, q):
    num_1 = 1 - np.exp((-1)*(p+q)*t)
    denom_1 = 1 + (q/p)*np.exp((-1)*(p+q)*t)
    num_2 = 1 - np.exp((-1)*(p+q)*(t-1))
    denom_2 = 1 + (q/p)*np.exp((-1)*(p+q)*(t-1))
    return 100*(num_1/denom_1) - 100*(num_2/denom_2)
x = df_4['t']
y = df_4['N(t)']
# running nonlinear regression
p_opt, p_cov = curve_fit(compute_N, x, y, p0 = [0.02, 0.5])
# extracting p and q from optimal parameters
p, q = p_opt[0], p_opt[1]
# display values
print(f'p = {round(p,3)}')
print(f'q = {round(q,3)}')
M = 100
# extrapolate
df_4 = extrapolate_cont_bass(df_4, p, q, M, 30)

# display value
print(f'N(30) = {round(df_4.loc[29, "N(t)"],3)}')
print()

print('QUESTION 2')
M = 100
p = [0.005,0.04,0.08]
q = np.linspace(0.1, 0.8, 3)
index = 0
for i, p_val in enumerate(p):
    for j, q_val in enumerate(q):
        df = pd.read_excel('adoptionseries2_with_noise.xlsx')
        df.loc[0, 'A(t)'] = 0
        for k in range(1, len(df)):
            df.loc[k, 'A(t)'] = df.loc[k-1, 'N(t)'] + df.loc[k-1, 'A(t)']

        # extrapolations
        df = extrapolate_discrete_bass(df, p_val, q_val, M, 50)

        # plots
        index +=1
        fig = plt.figure(figsize = (3,2.5))
        ax = fig.add_subplot()
        ax.plot(df['t'], df['N(t)'], color = 'r')
        ax.set_xlabel('t')
        ax.set_ylabel('N(t)')
        ax.set_ylim([0,25])
        ax.text(x=30, y=17, s=f'p = {round(p_val,3)},\nq = {round(q_val,3)}')
        plt.grid()
        plt.show()
print()

