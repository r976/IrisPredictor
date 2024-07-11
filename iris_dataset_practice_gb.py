#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# In[101]:


df = pd.read_csv('Iris.csv')
df = df.drop('Id', axis=1)
df
label_encoder = preprocessing.LabelEncoder()
df['Species'] = label_encoder.fit_transform(df['Species'])
X,y = df.drop('Species',axis=1),df['Species']


# In[130]:


# kf = KFold(n_splits=3,random_state=42,shuffle=True)
# for train_index,val_index in kf.split(X):
#     X_train,X_val = X.iloc[train_index],X.iloc[val_index],
#     y_train,y_val = y.iloc[train_index],y.iloc[val_index],
    
# X_train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[131]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[132]:


gradient_booster = GradientBoostingClassifier(n_estimators=500, learning_rate=1)
gradient_booster.fit(X_train,y_train)
# print(classification_report(y_val,gradient_booster.predict(X_val)))
y_pred = gradient_booster.predict(X_test)
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))


# In[133]:


gb_grid_param = {'learning_rate': [0.01, 0.05, 0.1, 1],
                 'n_estimators' : [10, 50, 100, 500, 1000],
                 'max_depth': [2, 5, 8, 11],
                 'max_features': [1,2]}
gb_cv = KFold(n_splits=5)
gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_grid_param, cv=gb_cv)
gb_grid.fit(X_train, y_train)
print('GB best Parameters:', gb_grid.best_estimator_)
print('GB best Score:', gb_grid.best_score_)


# In[ ]:




