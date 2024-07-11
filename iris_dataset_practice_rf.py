#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[3]:


df = pd.read_csv('Iris.csv')
df


# In[4]:


X = df.drop(["Id", "Species"], axis=1)
y= df["Species"]
X


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.20, random_state=42)


# In[7]:


classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)


# In[8]:


y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[9]:


predictions = classifier.predict([[6.2, 3, 1.7, 1.9]])
predictions

