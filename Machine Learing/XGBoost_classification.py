# XGBoost Classifiction

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset=pd.read_csv(r"C:\Users\divya\Downloads\Churn_Modelling.csv")

X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# to change the gender ton 1-0 form
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))


X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

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
cm
