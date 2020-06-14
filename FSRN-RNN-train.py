from __future__ import print_function

import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import cv2, os
import seaborn as sns
import pandas as pd 
import warnings
from scipy.misc import toimage, imsave
import glob
import gc
from sklearn.utils import shuffle
from scipy.linalg import norm
from scipy import sum, average
from scipy.interpolate import RectBivariateSpline

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import keras, sys, time
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import Adam, SGD
# from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
# from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model

from our_train_D2N_raytrace_datagen import DataGenerator
#list visible devices
from tensorflow.python.client import device_lib


TRAIN_NUM = 30600
VAL_NUM = 5400
BATCH_SIZE = 32
H = 128
W = 128


def FluidNet( nClasses, input_height=128, input_width=128):
    assert input_height%32 == 0
    assert input_width%32 == 0
    IMAGE_ORDERING =  "channels_last" 
    
    img_input = Input(shape=(3,input_height,input_width, 4), name='input_0')
    
    # Temporal Consistency
    x = ConvLSTM2D(filters=4, kernel_size=(2, 2)
                       , data_format=IMAGE_ORDERING 
                       , name='convLSTM_1'
                       , recurrent_activation='hard_sigmoid'
                       , activation='sigmoid'
                       , padding='same', return_sequences=True)(img_input)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=4, kernel_size=(2, 2)
                       , data_format=IMAGE_ORDERING
                       , name='convLSTM_2'
                       , recurrent_activation='hard_sigmoid'
                       , activation='sigmoid'
                       , padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(filters=4, kernel_size=(2, 2)
                       , data_format=IMAGE_ORDERING
                       , name='convLSTM_3'
                       , recurrent_activation='hard_sigmoid'
                       , activation='sigmoid'
                       , padding='same', return_sequences=False)(x)
    x = BatchNormalization()(x)

    o = x
    o = (Activation('sigmoid', name="refine_out"))(o)

    model = Model(img_input, o)
       
    return model

# Custom loss
def depth_loss (y_true, y_pred):  
    d = tf.subtract(y_pred,y_true)
    n_pixels = 128 * 128
    square_n_pixels = n_pixels * n_pixels
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d,1)
    sum_d = tf.reduce_sum(d,1)
    square_sum_d = tf.square(sum_d)
    mid_output = tf.reduce_mean((sum_square_d/n_pixels) - 0.5* (square_sum_d/square_n_pixels))

    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    paddings_y = tf.constant([[0,0],[1,0],[0,0],[0,0]])
    paddings_x = tf.constant([[0,0],[0,0],[1,0],[0,0]])
    
    pad_dy_true = tf.pad(dy_true, paddings_y, "CONSTANT")
    pad_dy_pred = tf.pad(dy_pred, paddings_y, "CONSTANT")
    pad_dx_true = tf.pad(dx_true, paddings_x, "CONSTANT")
    pad_dx_pred = tf.pad(dx_pred, paddings_x, "CONSTANT")

    pad_dy_true = pad_dy_true[:,:-1,:,:]
    pad_dy_pred = pad_dy_pred[:,:-1,:,:]
    pad_dx_true = pad_dx_true[:,:,:-1,:]
    pad_dx_pred = pad_dx_pred[:,:,:-1,:]

    term3 = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true) + K.abs(pad_dy_pred - pad_dy_true) + K.abs(pad_dx_pred - pad_dx_true), axis=-1)
    
    depth_output = mid_output + term3
    depth_output = K.mean(depth_output)
    return depth_output

def normal_loss(y_true, y_pred):
    d = tf.subtract(y_pred,y_true)
    n_pixels = 128 * 128
    square_n_pixels = n_pixels * n_pixels
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d,1)
    sum_d = tf.reduce_sum(d,1)
    square_sum_d = tf.square(sum_d)
    normal_output = tf.reduce_mean((sum_square_d/n_pixels) - 0.5* (square_sum_d/square_n_pixels))
    return normal_output

def depth_to_normal(y_pred_depth,y_true_normal, scale_true):
    Scale = 127.5

    depth_min = scale_true[:,0:1,0:1]
    depth_max = scale_true[:,0:1,1:2]
    
    normal_min = scale_true[:,0:1,2:3]  
    normal_max = scale_true[:,0:1,3:4]

    y_pred_depth = depth_min + (depth_max - depth_min) * y_pred_depth
    y_true_normal = normal_min + (normal_max - normal_min) * y_true_normal
    
    zy, zx = tf.image.image_gradients(y_pred_depth)
    
    zx = zx*Scale
    zy = zy*Scale
    
    normal_ori = tf.concat([zy, -zx, tf.ones_like(y_pred_depth)], 3) 
    new_normal = tf.sqrt(tf.square(zx) +  tf.square(zy) + 1)
    normal = normal_ori/new_normal
   
    normal += 1
    normal /= 2  
    return normal

def combined_loss(y_true,y_pred):

    depth_true = y_true[:,:,:,0]
    normal_true = y_true[:,:,:,1:4]
    scale_true = y_true[:,:,:,4:]

    depth_pred = y_pred[:,:,:,0]
    normal_pred = y_pred[:,:,:,1:]
    #scale_pred = y_pred[:,:,:,4:]

    depth_true = tf.expand_dims(depth_true, -1)
    depth_pred = tf.expand_dims(depth_pred, -1)

    alpha = 0.4
    beta = 0.4
    gamma = 0.2

    #depth loss
    loss_depth = alpha*(depth_loss(depth_true,depth_pred))

    #normal loss
    loss_normal = beta*(normal_loss(normal_true,normal_pred))#beta*mean_squared_error(normal_true,normal_pred)#beta*(normal_loss(normal_true,normal_pred))
    
    #normal from depth
    normal_from_depth = depth_to_normal(depth_pred,normal_true, scale_true)
    loss_depth_to_normal = gamma*(normal_loss(normal_true,normal_from_depth)) 

    return (loss_depth + loss_normal + loss_depth_to_normal)
    
def print_layer_trainable():
    for layer in model_depth.layers:
        print("{0}:\t{1}".format(layer.trainable,layer.name))

# TRAIN
with tf.device("/gpu:0"):

    model = FluidNet(nClasses     = 4, 
             input_height = 128, 
             input_width  = 128)
    
    model.summary()

    # Load post-scaled data predicted from FSRN-CNN
    gc.collect()
    
    X_train = np.load(dir_references+"X_FSRN_RNN_train_{}.npy".format(TRAIN_NUM))
    X_train = np.array(X_train)
    print(X_train.shape)
   
    
    y_train = np.load(dir_references+"Y_FSRN_RNN_train_{}.npy".format(TRAIN_NUM))
    y_train = np.array(y_train)   
    print(y_train.shape)

    X_test = np.load(dir_references+"X_FSRN_RNN_val_{}.npy".format(VAL_NUM))
    X_test = np.array(X_test)
    print(X_test.shape)
   
    
    y_test = np.load(dir_references+"Y_FSRN_RNN_val_{}.npy".format(VAL_NUM))
    y_test = np.array(y_test)   
    print(y_test.shape)

    #create model and train
    training_log = TensorBoard(log_folder)
    weight_filename = weight_folder + "pretrained_FSRN_RNN.h5"

    stopping = EarlyStopping(monitor='val_loss', patience=2)

    checkpoint = ModelCheckpoint(weight_filename,
                                 monitor = "val_loss"
                                 save_best_only = True,
                                 save_weights_only = True)
    #Plot loss
    dir_plot = "plot/" 
    
    model = FluidNet(nClasses     = 4,  
             input_height = 128, 
             input_width  = 128)
    
    model.summary()
    plot_model(model,to_file=dir_plot+'model_FSRN_RNN_.png',show_shapes=True)

    
    epochs = 15
    learning_rate = 0.001
    batch_size = BATCH_SIZE
    loss_funcs = {
        "refine_out": combined_loss,
    }
    loss_weights = {"refine_out": 1.0}
    
    print('lr:{},epoch:{},batch_size:{}'.format(learning_rate,epochs,batch_size))
    
    adam = Adam(lr=learning_rate)
    model.compile(loss= loss_funcs,
              optimizer=adam,
              loss_weights=loss_weights,
              metrics=["accuracy"])

    
    gc.collect()
    
    history = model.fit(X_train,y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2, 
                    validation_data=(X_test,y_test),
                    callbacks = [training_log,checkpoint])

    fig = plt.figure(figsize=(10,20)) 

    ax = fig.add_subplot(1,2,1)
    for key in ['loss', 'val_loss']:
        ax.plot(history.history[key],label=key)
    ax.legend()

    ax = fig.add_subplot(1,2,2)
    for key in ['acc', 'val_acc']:
        ax.plot(history.history[key],label=key)
    ax.legend()
    fig.savefig(dir_plot+"Loss_FSRN_RNN_"+str(epochs)+".png")   # save the figure to file
    plt.close(fig)