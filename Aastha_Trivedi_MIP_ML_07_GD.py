#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Loading Dataset

# In[2]:


data = pd.read_csv('Downloads/goldstock.csv')
data


# In[3]:


data.head()


# # Data Preprocessing

# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.dropna()


# In[8]:


data.drop_duplicates()


# In[9]:


data.corr()


# # Exploratory Data Analysis(EDA)

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


sns.pairplot(data)


# In[12]:


data.hist(figsize=(20,10),bins=50)


# In[13]:


data.corr()


# In[14]:


sns.heatmap(data.corr(),annot=True)


# # Splitting Dataset into Training And Testing

# In[15]:


X = data.drop(['Date','Volume'],axis=1)
Y = data['Volume']


# In[16]:


X


# In[17]:


Y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# In[20]:


X_train


# In[21]:


Y_train


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Linear Regression

# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[25]:


model = LinearRegression()
model.fit(X_train_scaled, Y_train)


# In[26]:


y_pred = model.predict(X_test_scaled)


# In[27]:


print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('R-squared:', r2_score(Y_test, y_pred))


# In[28]:


model.score(X_test_scaled, Y_test)


# In[29]:


model.score(X_train_scaled, Y_train)


# In[30]:


Y_test = list(Y_test)


# In[31]:


plt.plot(Y_test, color='yellow', label='Actual Value')
plt.plot(y_pred, color='green', label='Predicted Value')
plt.title('Actual vs predicted value')
plt.xlabel('Number of value')
plt.ylabel('gold')
plt.legend()
plt.show()


# # Prediction

# In[33]:


result = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
result


# # Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


model = RandomForestClassifier()
model.fit(X_train_scaled, Y_train)


# In[36]:


y_pred = model.predict(X_test_scaled)


# In[37]:


print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('R-squared:', r2_score(Y_test, y_pred))


# In[38]:


model.score(X_test_scaled, Y_test)


# In[39]:


model.score(X_train_scaled, Y_train)


# In[40]:


Y_test = list(Y_test)


# In[41]:


plt.plot(Y_test, color='magenta', label='Actual Value')
plt.plot(y_pred, color='green', label='Predicted Value')
plt.title('Actual vs predicted value')
plt.xlabel('Number of value')
plt.ylabel('gold')
plt.legend()
plt.show()


# # Prediction

# In[42]:


result = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
result


# In[43]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[44]:


print(accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))


# # Logistic Regression

# In[45]:


from sklearn.linear_model import LogisticRegression


# In[46]:


model = LogisticRegression()
model.fit(X_train_scaled, Y_train)


# In[47]:


y_pred = model.predict(X_test_scaled)


# In[48]:


print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('R-squared:', r2_score(Y_test, y_pred))


# In[49]:


model.score(X_test_scaled, Y_test)


# In[50]:


model.score(X_train_scaled, Y_train)


# In[51]:


Y_test = list(Y_test)


# In[52]:


plt.plot(Y_test, color='red', label='Actual Value')
plt.plot(y_pred, color='cyan', label='Predicted Value')
plt.title('Actual vs predicted value')
plt.xlabel('Number of value')
plt.ylabel('gold')
plt.legend()
plt.show()


# # Prediction

# In[53]:


result = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
result


# In[54]:


print(accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))


# # Decision Tree

# In[55]:


from sklearn.tree import DecisionTreeClassifier


# In[56]:


model = DecisionTreeClassifier()
model.fit(X_train_scaled, Y_train)


# In[57]:


y_pred = model.predict(X_test_scaled)


# In[58]:


print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('R-squared:', r2_score(Y_test, y_pred))


# In[59]:


model.score(X_test_scaled, Y_test)


# In[60]:


model.score(X_train_scaled, Y_train)


# In[61]:


Y_test = list(Y_test)


# In[62]:


plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(y_pred, color='green', label='Predicted Value')
plt.title('Actual vs predicted value')
plt.xlabel('Number of value')
plt.ylabel('gold')
plt.legend()
plt.show()


# # Prediction

# In[63]:


result = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
result


# In[64]:


print(accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))


# # SVC

# In[65]:


from sklearn.svm import SVC


# In[66]:


model = SVC()
model.fit(X_train_scaled, Y_train)


# In[67]:


y_pred = model.predict(X_test_scaled)


# In[68]:


print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('R-squared:', r2_score(Y_test, y_pred))


# In[69]:


model.score(X_test_scaled, Y_test)


# In[70]:


model.score(X_train_scaled, Y_train)


# In[71]:


Y_test = list(Y_test)


# In[72]:


plt.plot(Y_test, color='yellow', label='Actual Value')
plt.plot(y_pred, color='blue', label='Predicted Value')
plt.title('Actual vs predicted value')
plt.xlabel('Number of value')
plt.ylabel('gold')
plt.legend()
plt.show()


# # Prediction

# In[73]:


result = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
result


# In[74]:


print(accuracy_score(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))


# In[ ]:




