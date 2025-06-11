import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r"C:\Users\divya\Downloads\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

# Divide in ratio

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0) # We can only use train_size, or test_size other one will default change , random_state=0 it stop picking random data after picking first time

from sklearn.linear_model import LinearRegression

regressor= LinearRegression()   
regressor.fit(x_train, y_train)

#Prediction points

y_pred=regressor.predict(x_test)


#compare of actual and predicted

comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

#Visualization

plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regressor.predict(x_test),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#slope m
m_slope=regressor.coef_
print(m_slope)

# c - intercept

c_intercept=regressor.intercept_
print(c_intercept)

# new employee 12 year exp prediction result
y_12yrs= m_slope*12 +c_intercept

print(y_12yrs)

