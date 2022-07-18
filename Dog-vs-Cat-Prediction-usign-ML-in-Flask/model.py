#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data.csv")

X = df[["Height", "Weight"]]
y = df["Species"]

clf = GaussianNB() 
clf.fit(X, y)


# In[4]:


import joblib

joblib.dump(clf, "clf.pkl")


# In[ ]:




