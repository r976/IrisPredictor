#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[150]:


df = pd.read_csv('Iris.csv')
X = df.drop(columns=['Id', 'Species'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train) 
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
#0.8-1.0
predictions = model.predict([[6.2, 3, 1.7, 1.9]])
predictions
#score


# In[100]:


df

