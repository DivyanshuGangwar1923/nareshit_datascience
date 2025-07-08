#Naive Bayes Classification

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r"C:\Users\divya\Downloads\logit classification.csv")

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=0)

# feature scaling (normalization or scalar)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Gaussian NB
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()
#classifier.fit(X_train,y_train)

#Bernoulli NB
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
classifier.fit(X_train,y_train)

#Multi-Nomial NB
#from sklearn.naive_bayes import MultinomialNB
#classifier=MultinomialNB()
#classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)


bias=classifier.score(X_train,y_train)
variance=classifier.score(X_test,y_test)

print(bias)
print(variance)
print(ac)
