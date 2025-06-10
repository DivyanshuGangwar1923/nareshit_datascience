import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset=pd.read_csv(r'C:/Users/divya/Downloads/Data.csv')


x=dataset.iloc[:,:-1].values

y=dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer # Spyder 4(Transformers for missing value imputation.)

imputer= SimpleImputer()

imputer= imputer.fit(x[:,1:3])

x[:,1:3]=imputer.transform((x[:,1:3]))

# Convert categorical data to numerical data

from sklearn.preprocessing import LabelEncoder

labelencoder_x=LabelEncoder()   

labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])

labelencoder_y= LabelEncoder()  

y=labelencoder_y.fit_transform(y)

# Divide in ratio

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0) # We can only use train_size, or test_size other one will default change , random_state=0 it stop picking random data after picking first time
