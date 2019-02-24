
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import models
from tensorflow.keras import layers
keras.__version__


# In[4]:


data_10000 = pd.read_csv('training_fullclean.csv',nrows=400000)


# In[5]:


data_20000 = pd.read_csv('testing_fullclean.csv',nrows=400000)


# In[5]:


#rawData = pd.read_csv('datathon_propattributes.csv')
#slicedData = rawData[:100]
data = data_10000
#data = data.drop([])
#data['sale_amt'].describe()


# In[6]:


#Distribution of Property Sale Prices
plt.hist(data.sale_amt, facecolor = 'green')
plt.xlabel('Property Prices ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Property Sale Prices')
plt.show()


# In[8]:


#Distribution of Property Square Feet
plt.hist(data.building_square_feet, 50, facecolor = 'green')
plt.xlabel('Property Size (Square Feet)')
plt.ylabel('Frequency')
plt.title('Distribution of Property Sizes')
plt.show()


# In[9]:


#Distribution of Number of Bedrooms in Property
plt.hist(data.bedrooms, 50, facecolor = 'green')
plt.xlabel('Bedrooms (Units)')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Bedrooms in Property')
plt.show()


# In[10]:


#Distribution of Number of Bathrooms in Property
plt.hist(data.total_baths_calculated, 50, facecolor = 'green')
plt.xlabel('Bathroom (Units)')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Bathrooms in Property')
plt.show()


# In[11]:


#Scatter Plot of Number of Bedrooms and Bathrooms in Property
plt.scatter(data.bedrooms,data.total_baths_calculated,facecolor = 'green')
plt.xlabel('Bathroom (Units)')
plt.ylabel('Bedroom (Units)')
plt.title('Plot of Number of Bedrooms and Bathrooms in Property')
plt.show()


# # Neural Nets Section

# ## Train_val split

# In[6]:


data_10000['land_square_footage'].describe()


# In[12]:


y = data_10000.sale_amt
yy = data_20000.sale_amt
#X = data_10000[['prop_zip_code','land_square_footage',
#                           'building_square_feet','effective_year_built',
#                          'bedrooms','total_baths_calculated','air_conditioning_Yes',
#                           'condition_Excellent','heating_type_Yes']]
#df[((df.B - df.B.mean()) / df.B.std()).abs() < 3]
X = data_10000.drop(columns = ['sale_amt','stories_cd'])
XX = data_10000.drop(columns = ['sale_amt','stories_cd'])
#X.drop(X.columns[cols],axis=1,inplace=True)
#X = tf.keras.utils.normalize(X)
#X.head


# In[13]:


#list(X)
y = np.log1p(y)
X = np.log1p(X)
yy = np.log1p(yy)
XX = np.log1p(XX)


# In[16]:


X.isnull().values.any()


# In[17]:


XX.isnull().values.any()


# In[45]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, _, y_test, _ = train_test_split(XX, yy, test_size = 0, random_state=42)


# mean = X_train.mean(axis=0)
# X_train -= mean
# std = X_train.std(axis=0)
# X_train /= std
# 
# X_val -= mean
# X_val /= std

# In[49]:


#train_data.shape
X_train.shape
#y_train.shape


# In[50]:


X_train.shape[1]


# In[51]:


#val_data.shape
X_val.shape
#y_val.shape


# In[52]:


def build_model():
    model = models.Sequential()
    model.add(keras.layers.Dense(256, activation='relu',
                           input_shape=(X_train.shape[1],)))
    #model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    #model.add(keras.layers.Dense(256, activation='relu'))
    #model.add(keras.layers.Dense(256, activation='relu'))
    #model.add(keras.layers.Dense(256, activation='relu'))
    #model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    #model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


# In[53]:


# fix random seed
seed = 123
np.random.seed(seed)


# In[54]:


# Build the Keras model (already compiled)
model = build_model()
# Train the model (in silent mode, verbose=0)
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_val, y_val))


# In[55]:


model.summary()


# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# In[56]:


# Evaluate the model on the validation data
val_mse, val_mae = model.evaluate(X_val, y_val)


# In[57]:


results = model.evaluate(X_val, y_val)
results


# In[58]:


preds = model.predict(X_test)


# In[59]:


predss = [preds[i][0] for i in range(len(preds))]
predss = np.array(predss)


# In[60]:


predss_exp = np.exp(predss)+1


# In[61]:


predss_exp


# In[62]:


y_test = np.array(y_test)


# In[63]:


y_test_exp = np.exp(y_test) +1


# In[64]:


y_test


# In[65]:


accuracy = ((predss-y_test)/y_test)
#accuracy = accuracy_score(y_val, predss)
accuracy.mean()


# In[66]:


accuracy_exp = ((predss_exp-y_test_exp)/y_test_exp)
#accuracy = accuracy_score(y_val, predss)
accuracy_exp.mean()


# In[67]:


#evaluate your model on the test set
results = model.evaluate(X_test, y_test)


# In[68]:


#run this to see your mae
results[1]

