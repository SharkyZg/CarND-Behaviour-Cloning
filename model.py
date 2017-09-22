#Import packages required for the model training

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import math

from keras import initializations
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

import tensorflow as tf
tf.python.control_flow_ops = tf

from pathlib import Path
import json

# Imorting the driving log
csv_file = 'data/driving_log.csv'
driving_log = pd.read_csv(csv_file, header =0, index_col = False)

# Removing low speeds
ind = driving_log['throttle']>.20
driving_log= driving_log[ind].reset_index()


# Preparing steering data
smooth_steering = np.array(driving_log.steering,dtype=np.float32)
driving_log['steer_smooth'] = pd.Series(smooth_steering, index=driving_log.index)

#Determining imput image shape
image = cv2.imread(driving_log['center'][0].strip())
rows,cols,channels = image.shape

# Size in pixels to which imput image will be resized
input_image_size_col = 64
input_image_size_row = 64

# Cutting of irrelevant area of the image (skyline) and resizing image for model training.
def preprocess_image(image):
    shape = image.shape
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(input_image_size_col,input_image_size_row), interpolation=cv2.INTER_AREA)    
    return image 

# Augmenting brightness of images for driving in shadows or night
def augment_brightness_of_image(image):
    image_bright = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image_bright[:,:,2] = image_bright[:,:,2]*random_bright
    image_bright = cv2.cvtColor(image_bright,cv2.COLOR_HSV2RGB)
    return image_bright

# Shifting image by using warpAffine function
def transform_image(image,steer,trans_range):
    transform_x = trans_range*np.random.uniform()-trans_range/2
    steering_angle = steer + transform_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    transformation_matrix = np.float32([[1,0,transform_x],[0,1,tr_y]])
    image_transformed = cv2.warpAffine(image,transformation_matrix,(cols,rows))
    
    return image_transformed,steering_angle,transform_x

# Preprocessing image for model training
def preprocess_image_file_train(line_data):
    #Choosing one random camera side
    random_camera = np.random.randint(3)
    if (random_camera == 0):
        imported_image = line_data['left'][0].strip()
        shift_angle = .25
    if (random_camera == 1):
        imported_image = line_data['center'][0].strip()
        shift_angle = 0.
    if (random_camera == 2):
        imported_image = line_data['right'][0].strip()
        shift_angle = -.25
    swifted_steering_angle = line_data['steer_smooth'][0] + shift_angle
    image = cv2.imread(imported_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image,swifted_steering_angle,transform_x = transform_image(image,swifted_steering_angle,150)
    image = augment_brightness_of_image(image)
    image = preprocess_image(image)
    image = np.array(image)
    #randomly inverse image for additional augmentation
    randomly_inverse_image = np.random.randint(2)
    if randomly_inverse_image==1:
        image = cv2.flip(image,1)
        swifted_steering_angle = -swifted_steering_angle
        
    return image,swifted_steering_angle

# Preprocess image for predicting (no augmentation)
def preprocess_image_for_predicting(line_data):
    imported_image = line_data['center'][0].strip()
    image = cv2.imread(imported_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image = np.array(image)
    return image

# Random batch generator of input images and steering angles
def train_data_batch_generator(data, batch_size):
    batch_images = np.zeros((batch_size, input_image_size_row, input_image_size_col, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            x,y = preprocess_image_file_train(line_data)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering

# Generate input images and steering angles for validation
def generate_validation_data(data):
    while 1:
        for i_line in range(len(data)):
            line_data = data.iloc[[i_line]].reset_index()
            x = preprocess_image_for_predicting(data)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = line_data['steer_smooth'][0]
            y = np.array([[y]])
            yield x, y
            
# Save keras model to disk
def save_keras_model(file_model_json,file_weights):
    if Path(file_model_json).is_file():
        os.remove(file_model_json)
    json_string = model.to_json()
    with open(file_model_json,'w' ) as f:
        json.dump(json_string, f)
    if Path(file_weights).is_file():
        os.remove(file_weights)
    model.save_weights(file_weights)



#NVIDIA neural network model
input_shape = (input_image_size_row, input_image_size_col, 3)
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1. , input_shape = input_shape))
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(1164, init='he_normal'))
model.add(ELU())
model.add(Dense(100, init='he_normal'))    
model.add(ELU())
model.add(Dense(50, init='he_normal'))
model.add(ELU())
model.add(Dense(10, init='he_normal'))
model.add(ELU())
model.add(Dense(1, init='he_normal'))

# Adam optimizer
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam,loss='mse')

batch_size = 256

validation_data_generator = generate_validation_data(driving_log)

# Train model
for i_model_number in range(10):

    train_data_generator = train_data_batch_generator(driving_log,batch_size)

    history = model.fit_generator(train_data_generator,
            samples_per_epoch=20224, nb_epoch=1,validation_data=validation_data_generator,
                        nb_val_samples=len(driving_log))
    
    file_model_json = 'model_' + str(i_model_number) + '.json'
    file_weights = 'model_' + str(i_model_number) + '.h5'
    
    save_keras_model(file_model_json,file_weights)
    
    val_loss = history.history['val_loss'][0]