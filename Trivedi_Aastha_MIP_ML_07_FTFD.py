#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # Load Data

# In[2]:


data = pd.read_csv('Downloads/FastagFraudDetection.csv')
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


data.Fraud_indicator.value_counts()


# In[10]:


h=data.columns
h


# # Exploratory Data Analysis(EDA)

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


sns.pairplot(data)


# In[13]:


data.hist()


# In[14]:


data.corr()


# In[15]:


sns.heatmap(data.corr(),annot=True)


# In[16]:


data['Timestamp']= pd.to_datetime(data['Timestamp'])
data['Timestamp'].info()


# In[17]:


sns.countplot(x='Fraud_indicator', data=data)
plt.show()


# In[18]:


sns.countplot(data= data, x='Vehicle_Speed', hue='Fraud_indicator', palette= ['Red', 'Green'])
plt.title('Speed Fraudulent Activity')
label=['Fraud', 'No-Fraud']
plt.show()


# In[19]:


sns.barplot(x='Vehicle_Type',y='Transaction_ID',data=data,hue='Fraud_indicator')
plt.legend(loc='upper right')

plt.show()


# In[20]:


data['Fraud_indicator'] = data['Fraud_indicator'].astype('category')
data['Fraud_indicator'] = pd.get_dummies(data['Fraud_indicator'],drop_first=True)
pd.get_dummies(data['Fraud_indicator'],drop_first=False)


# In[21]:


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# In[22]:


categorical_columns = ['Vehicle_Type', 'FastagID', 'TollBoothID', 'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location', 'Vehicle_Plate_Number', 'Fraud_indicator']
lb= LabelEncoder()

for col in categorical_columns:
    data[col]= lb.fit_transform(data[col].astype(str))


# In[23]:


data


# In[24]:


data=data.drop('Timestamp', axis=1)


# # Training and Testing Data

# In[25]:


X= data.drop('Fraud_indicator', axis=1)
y= data['Fraud_indicator']

X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=20)


# In[26]:


sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# # Random Forest

# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[29]:


pred = model.predict(X_test)


# In[30]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[31]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[33]:


cm = confusion_matrix(y_test,pred)
cm
sns.heatmap(cm,annot=True)


# In[34]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # Decision Tree

# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[37]:


pred = model.predict(X_test)


# In[38]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[39]:


cm1 = confusion_matrix(y_test,pred)
cm1
sns.heatmap(cm1,annot=True)


# In[40]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


model = LogisticRegression()
model.fit(X_train,y_train)


# In[43]:


pred = model.predict(X_test)


# In[44]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[45]:


cm2 = confusion_matrix(y_test,pred)
cm2
sns.heatmap(cm2,annot=True)


# In[46]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # KNN

# In[47]:


from sklearn.neighbors import KNeighborsClassifier


# In[48]:


model = KNeighborsClassifier()
model.fit(X_train,y_train)


# In[49]:


pred = model.predict(X_test)


# In[50]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[51]:


cm3 = confusion_matrix(y_test,pred)
cm3
sns.heatmap(cm3,annot=True)


# In[52]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # SVC

# In[53]:


from sklearn.svm import SVC


# In[54]:


model = SVC()
model.fit(X_train,y_train)


# In[55]:


pred = model.predict(X_test)


# In[56]:


print("Accuracy:",accuracy_score(y_test,pred))
print("*"*50)
print(confusion_matrix(y_test,pred))
print("*"*50)
print(classification_report(y_test,pred))


# In[57]:


cm4 = confusion_matrix(y_test,pred)
cm4
sns.heatmap(cm4,annot=True)


# In[58]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # Accuracy Plotting

# In[59]:


acc=[98.88,99.68,97.36,87.92,95.84]
name=['random','disi','log_re','KNN','SVC']
batch_size=[16,32,64,128,135]


# In[60]:


fig = plt.figure(figsize = (10, 5))
plt.rc('font', size=20)
plt.ylim((0,100))
plt.bar(name, acc,color=['blue','green','red','yellow','black'],width = 0.8,edgecolor='black')
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.show()


# In[61]:


plt.plot(batch_size,acc,'b-o',label='Accuracy over batch size for 1000 iterations');


# # Calibration Plot

# In[62]:


import scikitplot as skplt


# In[63]:


#CALIBRATION CURVE FOR ALL MODELS
rf_probas = RandomForestClassifier().fit(X_train, y_train).predict_proba(X_test)
dt_probas = DecisionTreeClassifier().fit(X_train, y_train).predict_proba(X_test)
lr_probas = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)
#svc_probas = SVC().fit(X_train,y_train).predict_proba(X_test)
Knn_probas = KNeighborsClassifier().fit(X_train,y_train).predict_proba(X_test)


# In[64]:


probas_list = [rf_probas, dt_probas,lr_probas,Knn_probas]
clf_names = ['Random Forest', 'Decision Tree','Logistic Regression', 'Knn']


# In[65]:


skplt.metrics.plot_calibration_curve(y_test,
                                     probas_list,
                                     clf_names, n_bins=15,
                                     figsize=(12,6)
                                     );


# In[ ]:




