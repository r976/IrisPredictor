#!/usr/bin/env python
# coding: utf-8

# In[203]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for data visualization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[204]:


df = pd.read_csv('Iris.csv')
df


# In[205]:


X = df.drop(['Id', 'Species'], axis=1)
y = df['Species']


# In[206]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[207]:


for df1 in [X_train, X_test]:
    for col in X_train.columns:
        col_median=X_train[col].median()
        df1[col].fillna(col_median, inplace=True)  
cols = X_train.columns


# In[208]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[209]:


X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.head()


# In[222]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[217]:


y_pred = knn.predict(X_test)

y_pred


# In[218]:


print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[219]:


print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))


# In[220]:


y_test.value_counts()


# In[221]:


null_accuracy = (13/(13+11+6))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))


# In[223]:


predictions = knn.predict([[6.2, 3, 1.7, 1.9]])
predictions

