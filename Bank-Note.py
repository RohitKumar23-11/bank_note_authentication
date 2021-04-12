#!/usr/bin/env python
# coding: utf-8

# # Import the libraries and methods

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.express as px


# ## Import the data set

# In[2]:


data = pd.read_csv("D:\Python notes\python program\BankNote_Authentication.csv")


# ## Exploratory Data Analysis

# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# ## Data Visualization

# In[7]:


col = ['r','c']
sns.countplot(x=data['class'],palette=col)


# In[8]:


cmap = sns.diverging_palette(300,175,s=40,l=75,n=10)
corrmat = data.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corrmat, cmap = cmap, annot=True, square = True)


# In[9]:


fig = px.scatter(data, 'variance', y = 'skewness', title='relationship between variance and skewness')
fig.show()


# In[10]:


fig = px.scatter(data, x = 'variance', y = 'curtosis', title='relationship between variance and curtosis')
fig.show()


# In[11]:


fig = px.scatter(data, x = 'variance', y = 'entropy', title='relationship between variance and entropy')
fig.show()


# In[12]:


fig = px.scatter(data, x = 'skewness', y = 'curtosis', title='relationship between skewness and curtosis')
fig.show()


# In[13]:


fig = px.scatter(data, x = 'skewness', y = 'entropy', title='relationship between skewness and entropy')
fig.show()


# In[14]:


fig = px.scatter(data, x = 'curtosis', y = 'entropy', title='relationship between curtosis and entropy')
fig.show()


# In[15]:


plt.hist(data['entropy'], density=True)
plt.hist(data['curtosis'], density=True)
plt.show()


# In[16]:


plt.hist(data['entropy'], density=True)
plt.hist(data['skewness'], density=True)
plt.show()


# In[17]:


plt.hist(data['entropy'], density=True)
plt.hist(data['variance'], density=True)
plt.show()


# In[18]:


plt.hist(data['curtosis'], density=True)
plt.hist(data['skewness'], density=True)
plt.show()


# In[19]:


plt.hist(data['curtosis'], density=True)
plt.hist(data['variance'], density=True)
plt.show()


# In[20]:


plt.hist(data['skewness'], density=True)
plt.hist(data['variance'], density=True)
plt.show()


# In[21]:


plt.hist(data['variance'], density=True)
plt.hist(data['skewness'], density=True)
plt.hist(data['curtosis'], density=True)
plt.hist(data['entropy'], density=True)
plt.show()


# In[22]:


cols= ["#6daa9f","#774571"]
feature = ['variance','skewness','curtosis','entropy']
for i in feature:
    plt.figure(figsize=(8,8))
    sns.swarmplot(x=data["class"], y=data[i], color="black", alpha=0.5)
    sns.boxenplot(x=data["class"], y=data[i], palette=cols)
    plt.show()


# # Data Preprocessing

# In[23]:


X = data.drop(['class'], axis=1)
y = data['class']


# In[24]:


X.head()


# In[25]:


X.info()


# In[26]:


X.describe()


# In[27]:


#Set up a standard scaler for the features
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df= s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)   
X_df.describe().T


# ## importing the model and methods for traning,compiling and testing the data 

# In[28]:


from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# In[29]:


#looking at the scaled features
colours =["#774571","#b398af","#f1f1f1" ,"#afcdc7", "#6daa9f"]
plt.figure(figsize=(20,10))
sns.boxenplot(data = X_df,palette = colours)
plt.xticks(rotation=90)
plt.show()


# In[30]:


#spliting test and training sets
X_train, X_test, y_train,y_test = train_test_split(X_df,y,test_size=0.2,random_state=42,stratify=y)


# In[42]:


model = Sequential()
# layers
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[43]:


# Train the ANN
history = model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_split=0.2)


# In[44]:


model.summary()


# ##  Checking the data accuracy, loss, f1-score, and visualization these

# In[45]:


val_accuracy = np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))


# In[46]:


# Predicting the test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
np.set_printoptions()


# In[47]:


# confusion matrix
cmap1 = sns.diverging_palette(275,150,  s=40, l=65, n=6)
plt.subplots(figsize=(12,8))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})


# In[48]:


print(classification_report(y_test, y_pred))


# In[49]:


plt.subplots(figsize=(12,8))
acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1,101)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[50]:


plt.subplots(figsize=(12,8))
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,101)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[51]:


plt.subplots(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[52]:


plt.subplots(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




