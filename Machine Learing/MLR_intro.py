import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\divya\Downloads\Investment.csv")

X=dataset.iloc[:,:-1]

y=dataset.iloc[:,4]

X=pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()    

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)

bias=regressor.score(X_train,y_train)
variance=regressor.score(X_test,y_test)

# print(bias,variance)

# now we build MLR(multi linear regression)

X=np.append(arr=np.ones((50,1)).astype(int), values=X,axis=1)

import statsmodels.api as sm

X_opt= X[:,[0,1,2,3,4,5]]

#Ordinary Least Squares

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() 

# Eliminate the index which has p >0.05 i.e 4

import statsmodels.api as sm

X_opt= X[:,[0,1,2,3,5]]

#Ordinary Least Squares

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() 

# Eliminate the index which has p >0.05 i.e 5

import statsmodels.api as sm

X_opt= X[:,[0,1,2,3]]

#Ordinary Least Squares

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() 

# Eliminate the index which has p >0.05 i.e 4

import statsmodels.api as sm

X_opt= X[:,[0,1,3]]

#Ordinary Least Squares

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() 

# Eliminate the index which has p >0.05 i.e 2



import statsmodels.api as sm

X_opt= X[:,[0,1]]

#Ordinary Least Squares

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary() 

# Eliminate the index which has p >0.05 i.e 3
# Digital marketing is the best fit 