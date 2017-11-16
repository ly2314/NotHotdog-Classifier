import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.contrib.layers import flatten
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import glob

import os

from utils import preprocessData, loadBlurImg

def getClassSize():
    cars = glob.glob('./images/car/**/*.jpg', recursive=True)
    notcars = glob.glob('./images/not-car/**/*.jpg', recursive=True)
    c = len(cars)
    n = len(notcars)
    if c > n:
        return c
    return n

def loadImgClass(classPath, classLable, classSize, imgSize):
    x = []
    y = []
    
    for path in classPath:
        img = loadBlurImg(path, imgSize)
        if img is None:
            continue
        x.append(img)
        y.append(classLable)
        
    while len(x) < classSize:
        randIdx = np.random.randint(0, len(classPath))
        img = loadBlurImg(classPath[randIdx], imgSize)
        if img is None:
            continue
        x.append(img)
        y.append(classLable)
        
    return x, y

def loadData(img_size, classSize):
    cars = glob.glob('./images/car/**/*.jpg', recursive=True)
    notcars = glob.glob('./images/not-car/**/*.jpg', recursive=True)    
    
    imgSize = (img_size, img_size)
    xCar, yCar = loadImgClass(cars, 0, classSize, imgSize)
    xNotCar, yNotCar = loadImgClass(notcars, 1, classSize, imgSize)
    print("There are", len(xCar), "car images")
    print("There are", len(xNotCar), "not car images")
    
    X = np.array(xCar + xNotCar)
    y = np.array(yCar + yNotCar)
    #y = y.reshape(y.shape + (1,))
    return X, y


def buildNetwork(X, keepProb):
    mu = 0
    sigma = 0.3
    
    output_depth = {
        0 : 3,
        1 : 8,
        2 : 16,
        3 : 32,
        4 : 3200,
        5 : 240,
        6 : 120, 
        7 : 43,
    }
    
    #Layer 1: Convolutional + MaxPooling + ReLu + dropout. Input = 64x64x3. Output = 30x30x8.
    layer_1 = tf.Variable( tf.truncated_normal([5,5,output_depth[0],output_depth[1]], mean=mu, stddev=sigma))
    layer_1 = tf.nn.conv2d(X, filter=layer_1, strides=[1,1,1,1], padding ='VALID')
    layer_1 = tf.add(layer_1, tf.zeros(output_depth[1]))
    layer_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer_1 = tf.nn.dropout(layer_1, keepProb)
    layer_1 = tf.nn.relu(layer_1)
    
    return layer_1

def karasModel(inputShape):
    model = Sequential()
    model.add(Convolution2D(8, 5, 5, border_mode='valid', input_shape=inputShape))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 3))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(240))

    model.add(Activation('relu'))
    model.add(Dense(120))

    #model.add(Activation('relu'))
    model.add(Dense(2))

    model.add(Activation('softmax'))
    return model 
   
size = 64
classSize = getClassSize()
scaled_X, y = loadData(size, classSize)

n_classes = len(np.unique(y))
print("Number of classes =", n_classes)

scaled_X = preprocessData(scaled_X)
#scaled_X = normalizeImages(scaled_X)
#scaled_X = normalizeImages2(scaled_X)
label_binarizer = LabelBinarizer()

#y = label_binarizer.fit_transform(y)
from keras.utils.np_utils import to_categorical
y = to_categorical(y)
print("y shape", y.shape)
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

print("train shape X", X_train.shape)
print("train shape y", y_train.shape)
inputShape = (size, size, 1)
model = karasModel(inputShape)

#y_one_hot = label_binarizer.fit_transform(y_train)
#y_one_hoy = tf.one_hot(y_train, 2)
print("train shape y", y.shape)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_train, y_train, nb_epoch=30, validation_split=0.1)

y_one_hot_test = label_binarizer.fit_transform(y_test)

metrics = model.evaluate(X_test, y_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))

model.save('car.h5')