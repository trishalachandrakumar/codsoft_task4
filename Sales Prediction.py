#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('D:/Trishala/CodSoft/Sales/carpurchasing.csv', encoding='latin-1')
df.rename(columns={'car purchase amount':'sales','credit card debt':'creditcarddebt','annual Salary':'annualSalary','net worth':'networth'},inplace=True)
df.head()


# In[3]:


df.groupby(['country'])['sales'].count().head(10).plot.bar()


# In[4]:


df.groupby(['gender'])['sales'].count().head(10).plot.hist()


# In[5]:


df.groupby(['age'])['sales'].count().head(10).plot.pie()


# In[6]:


df.annualSalary.plot.hist()


# In[7]:


df.creditcarddebt.plot.box()


# In[8]:


df.networth.plot()


# In[9]:


df.corr
plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr(), annot=True, cmap='copper')


# In[10]:


df['networth'] = pd.to_numeric(df['networth'], errors='coerce')
df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
df['creditcarddebt'] = pd.to_numeric(df['creditcarddebt'], errors='coerce')
df['annualSalary'] = pd.to_numeric(df['annualSalary'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['gender'] = pd.to_numeric(df['gender'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['customer e-mail'] = pd.to_numeric(df['customer e-mail'], errors='coerce')
df['customer name'] = pd.to_numeric(df['customer name'], errors='coerce')


# In[11]:


x = np.array(df.drop(['sales','customer name','customer e-mail','country'], axis=1))
y = np.array(df.sales)


# In[12]:


x_test, x_train, y_test, y_train = train_test_split(x,y,test_size=0.2)


# In[13]:


model = LinearRegression()


# In[14]:


model.fit(x_train,y_train)


# In[15]:


y_pred = model.predict(x_test)


# In[16]:


comp = pd.DataFrame({'Actual Values':y_test, 'Predicted Values':y_pred})
comp


# In[17]:


plt.scatter(y_test,y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid()
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)], color='red')
plt.title('Actual Sales V/S Predicted Sales')


# In[18]:


r2_score(y_test, y_pred)

