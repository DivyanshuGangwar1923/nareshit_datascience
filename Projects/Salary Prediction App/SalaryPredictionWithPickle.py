import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

dataset=pd.read_csv(r"C:\Users\divya\Downloads\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

# Divide in ratio
x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0) # We can only use train_size, or test_size other one will default change , random_state=0 it stop picking random data after picking first time


#Train the model
regressor= LinearRegression()   
regressor.fit(x_train, y_train)

#Prediction points

y_pred=regressor.predict(x_test)


#compare of actual and predicted

comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

#Bias and Variance score
bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_test,y_test)
print(variance)

#Visualization
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for a new employee with 12 years of experience
y_12yrs_pred = regressor.predict([[12]])[0]
print(f'After 12 year expression employee will get {y_12yrs_pred}')

# Predict salary for a new employee with 25 years of experience
y_25yrs_pred = regressor.predict([[25]])[0]
print(f'After 25 year expression employee will get {y_25yrs_pred}')


filename='linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print('Model has been pickled and saved as linear_regression_model.pkl')

print(os.getcwd())


