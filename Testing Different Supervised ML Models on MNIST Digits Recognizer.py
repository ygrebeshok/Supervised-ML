#!/usr/bin/env python
# coding: utf-8

# # Testing Different Supervised ML Models on MNIST Digits Recognizer

# In[10]:


import pandas as pd
import numpy as np

df_train = pd.read_csv('/Users/ulia/digit-recognizer/train.csv')
df_test = pd.read_csv('/Users/ulia/digit-recognizer/test.csv')

# Shuffle Data
np.random.seed(123)
df_train = df_train.iloc[np.random.permutation(len(df_train))]
df_train.head()


# In[12]:


# Get pixels columns named 'pixel1', 'pixel2', ..., 'pixel784'
pixel_columns = ['pixel' + str(i) for i in range(784)]


# ## KNN

# In[13]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Assuming df_train has both features and labels
X_train = df_train[pixel_columns].values
y_train = df_train['label'].values

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=100)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
X_test = scaler.transform(df_test.values)
predictions = knn_model.predict(X_test)

# Format predictions for submission
submission_df = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission_df.to_csv('submission-knn.csv', index=False)


# ## SVM

# In[14]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Assuming df_train has both features and labels
X_train = df_train[pixel_columns].values
y_train = df_train['label'].values

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the SVM model
svm_model = SVC(kernel='linear', C=10.0)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
X_test = scaler.transform(df_test.values)
predictions = svm_model.predict(X_test)

# Format predictions for submission
submission_df = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission_df.to_csv('submission-svm.csv', index=False)


# ## Logistic Regression

# In[15]:


# Reshape the data
y_train = df_train['label']
X_train = df_train[pixel_columns].values.reshape(-1, 28 * 28)
X_test = df_test.values.reshape(-1, 28 * 28)


# In[16]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create a pipeline with normalization and logistic regression
model = Pipeline([
    ('scaler', StandardScaler()),  # Normalization
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit the model to your training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# In[17]:


submission_df = pd.DataFrame({
    'ImageId': df_test.index + 1,
    'Label': y_pred
})

submission_df.to_csv('submission-logistic.csv', index=False)


# ## CNN

# In[18]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import cv2 as cv

from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras import models
from keras.optimizers import Adam,RMSprop 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import pickle


# In[19]:


sample_size = df_train.shape[0]
validation_size = int(df_train.shape[0]*0.1) # validation 10% 

# train
train_x = df_train.iloc[:sample_size - validation_size, 1:].to_numpy().reshape([sample_size - validation_size, 28, 28, 1])
train_y = df_train.iloc[:sample_size - validation_size, 0].to_numpy().reshape([sample_size - validation_size, 1])

# validation
valid_x = df_train.iloc[sample_size - validation_size:, 1:].to_numpy().reshape([validation_size, 28, 28, 1])
valid_y = df_train.iloc[sample_size - validation_size:, 0].to_numpy().reshape([validation_size, 1])


# In[20]:


test_x = df_test.iloc[:,:].to_numpy().reshape([-1,28,28,1])


# In[22]:


# Normalizing data
train_x = train_x/255
valid_x = valid_x/255
test_x = test_x/255


# In[27]:


import seaborn as sns
sns.set_theme()


# In[28]:


# Cheacking frequency of digits in trainingset
counts = df_train.iloc[:sample_size-validation_size,:].groupby('label')['label'].count()

# Counts
f = plt.figure(figsize=(8,4))
f.add_subplot(111)

plt.bar(counts.index,counts.values,width = 0.8,color="#5688fc")
for i in counts.index:
    plt.text(i,counts.values[i]+50,str(counts.values[i]),horizontalalignment='center',fontsize=8)

plt.tick_params(labelsize = 12)
plt.xticks(counts.index)
plt.xlabel("Digits",fontsize=12)
plt.ylabel("Frequency",fontsize=12)
plt.title("Frequency Graph for Training set",fontsize=14) 
plt.show()


# In[29]:


# Cheacking frequency of digits in validation set
counts = df_train.iloc[sample_size-validation_size:,:].groupby('label')['label'].count()

# counts
f = plt.figure(figsize=(8,4))
f.add_subplot(111)

plt.bar(counts.index,counts.values,width = 0.8,color="#5688fc")
for i in counts.index:
    plt.text(i,counts.values[i]+5,str(counts.values[i]),horizontalalignment='center',fontsize=8)

plt.tick_params(labelsize = 12)
plt.xticks(counts.index)
plt.xlabel("Digits",fontsize=12)
plt.ylabel("Frequency",fontsize=12)
plt.title("Frequency Graph for Validation set",fontsize=14)
plt.show()


# In[30]:


model = models.Sequential()


# In[31]:


# Block 1
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())
model.add(Conv2D(32,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation="sigmoid"))


# In[32]:


initial_lr = 0.001
loss = "sparse_categorical_crossentropy"
model.compile(Adam(learning_rate=initial_lr), loss=loss ,metrics=['accuracy'])
model.summary()


# In[34]:


epochs = 20
batch_size = 256
history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[valid_x,valid_y])


# In[35]:


model.save("model.h5")
with open('history_1.hs', 'wb') as history:
    pickle.dump(history_1,history)

# Diffining Figure
f = plt.figure(figsize=(20,7))

#Adding Subplot 1 (For Accuracy)
f.add_subplot(121)

plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()


# In[39]:


val_p = np.argmax(model.predict(valid_x),axis =1)

error = 0
confusion_matrix = np.zeros([10,10])
for i in range(valid_x.shape[0]):
    confusion_matrix[valid_y[i],val_p[i]] += 1
    if valid_y[i]!=val_p[i]:
        error +=1
        
print("Confusion Matrix: \n\n" ,confusion_matrix)
print("\nErrors in validation set: " ,error)
print("\nError Percentage : " ,(error*100)/val_p.shape[0])
print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])
print("\nValidation set shape :",val_p.shape[0])


# In[40]:


#Augmentation with Keras
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(train_x)


# In[42]:


lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=1,factor=0.5, min_lr=0.00001)
epochs = 20
history_2 = model.fit(datagen.flow(train_x,train_y, batch_size=batch_size),steps_per_epoch=int(train_x.shape[0]/batch_size)+1,epochs=epochs,validation_data=[valid_x,valid_y],callbacks=[lrr])


# In[43]:


model.save("model_img_augmentation.h5")
with open('history_2.hs', 'wb') as history:
    pickle.dump(history_2,history)


# In[44]:


# Defining Figure
f = plt.figure(figsize=(20,7))
f.add_subplot(121)

#Adding Subplot 1 (For Accuracy)
plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['accuracy']+history_2.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_accuracy']+history_2.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()


#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['loss']+history_2.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch+list(np.asarray(history_2.epoch) + len(history_1.epoch)),history_1.history['val_loss']+history_2.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()


# In[46]:


val_p = np.argmax(model.predict(valid_x),axis =1)

error = 0
confusion_matrix = np.zeros([10,10])
for i in range(valid_x.shape[0]):
    confusion_matrix[valid_y[i],val_p[i]] += 1
    if valid_y[i]!=val_p[i]:
        error +=1
        
confusion_matrix,error,(error*100)/val_p.shape[0],100-(error*100)/val_p.shape[0],val_p.shape[0]

print("Confusion Matrix: \n\n" ,confusion_matrix)
print("\nErrors in validation set: " ,error)
print("\nError Percentage : " ,(error*100)/val_p.shape[0])
print("\nAccuracy : " ,100-(error*100)/val_p.shape[0])
print("\nValidation set shape :",val_p.shape[0])


# In[47]:


test_y = np.argmax(model.predict(test_x),axis =1)


# In[48]:


df_submission = pd.DataFrame([df_test.index+1,test_y],["ImageId","Label"]).transpose()
df_submission.to_csv("submission-cnn.csv",index=False)


# In[ ]:




