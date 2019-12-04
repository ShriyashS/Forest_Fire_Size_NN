# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:08:08 2019

@author: Shriyash Shende

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras import models
from keras import layers
from keras.layers import Activation, Dense

from sklearn.model_selection import train_test_split

fire = pd.read_csv('C:\\Users\\Good Guys\\Desktop\\pRACTICE\\EXCELR PYTHON\\Assignment\\Neurral Net\\forestfires.csv')
fire.info()
fire.drop(['month','day'], axis = 1, inplace = True)
fire.columns
s = fire.describe()



fire.loc[fire.size_category =="small","size_category"] = 0
fire.loc[fire.size_category =="large","size_category"] = 1

fire.size_category.value_counts().plot(kind="bar")

X = fire.drop(['size_category'], axis = 1)
Y = fire['size_category']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#Model

model = models.Sequential()
model.add(layers.Dense(64, activation='relu',input_dim = 28))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])

model.fit(np.array(X_train),np.array(Y_train),batch_size=512,validation_data=(X_test, Y_test),epochs=900)
pred_train = model.predict(np.array(X_test))
pred_train = pd.Series([i[0] for i in pred_train])

pred_train[[i < 0.34 for i in pred_train]] = 0
pred_train[[i > 0 for i in pred_train]] = 1


from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,pred_train)

print(cn)
per = cn[0,0] + cn[1,1]
p = per + cn[0,1] + cn[1,0]
per / p