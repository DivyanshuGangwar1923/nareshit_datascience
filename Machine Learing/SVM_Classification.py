#SVM Classifier

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

from sklearn.svm import SVC
classifier= SVC()   
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)

#Bias and Variance
bias=classifier.score(X_train,y_train)
variance=classifier.score(X_test,y_test)

print(bias)
print(cm)
print(variance)
print(ac)

### we need to pass future records to predict 

dataset1=pd.read_csv(r"C:\Users\divya\Downloads\final1.csv")

d2=dataset1.copy()

dataset1=dataset1.iloc[:,[3,4]].values

M=sc.fit_transform(dataset1)

y_pred1=pd.DataFrame()

d2['y_pred1']=classifier.predict(M)

d2.to_csv('final1.csv')
