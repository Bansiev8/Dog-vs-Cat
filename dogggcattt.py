#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle


# In[2]:


directory = r'G:\condapython\dogcat\finalset'
category = ['cats','dogs']


# In[3]:


img_size = 100
data = []
label = []
for cate in category:
    folder = os.path.join(directory, cate)
    label = category.index(cate)
    #print(folder)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(img_size,img_size))
        data.append([img_arr, label])  
        #print(img_arr)
        #break


# In[4]:


random.shuffle(data)


# In[5]:


x = [] #for features
y = [] #for labels
for features, label in data:
    x.append(features)
    y.append(label)
#print(y)


# In[6]:


x = np.array(x)
y = np.array(y)
x = x/255


# In[7]:


#pickle.dump(x,open('x.pkl','wb'))
#pickle.dump(y,open('y.pkl','wb'))


# In[10]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[11]:


model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())


model.add(Dense(128, input_shape = x.shape[1:], activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation = 'softmax'))


# In[12]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[13]:


model.fit(x, y, epochs = 10, validation_split = 0.3)


# In[ ]:




