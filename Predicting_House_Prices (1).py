#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = {
    'bedrooms': [2, 3, 4, 3, 5, 2, 4, 3, 5, 4, 2, 3, 3, 4, 5],
    'square_footage': [1200, 1500, 1800, 1600, 2400, 1000, 2200, 1400, 2600, 2000, 1100, 1350, 1700, 2100, 2500],
    'price': [200000, 250000, 300000, 270000, 400000, 180000, 350000, 230000, 450000, 320000, 190000, 220000, 280000, 330000, 410000]
}
df = pd.DataFrame(data)


# In[10]:


print(df)


# In[3]:


X = df[['bedrooms', 'square_footage']] 
y = df['price'] 


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[6]:


y_pred = model.predict(X_test)


# In[7]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[8]:


print("Mean Squared Error:", mse)
print("R^2 Score:", r2)


# In[9]:


new_house = [[3, 1800]] 
predicted_price = model.predict(new_house)
print("Predicted Price for new house:", predicted_price[0])


# In[ ]:




