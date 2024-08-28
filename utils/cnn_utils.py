# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:14:47 2022

@author: Valentin Kapitany
"""
#%%
import tensorflow as tf
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import random
import numpy as np
import time
import keras.backend as K
from tqdm import tqdm

#%%
class Architectures(object):
    def __init__(self,window_shape):
        self.window_shape = window_shape #shape of the local window
    def v6(self):
        """intensity window input, and position input. One hot encoding. Do not change"""
        
        dims = (self.window_shape[0], self.window_shape[1], 2)
        window_inp = tf.keras.layers.Input(shape=dims)
        window = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3,3), strides = 1, kernel_initializer='glorot_uniform',use_bias=False,
        )(window_inp)
        window = tf.keras.layers.LeakyReLU()(window)
        window = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3,3), strides = 1, kernel_initializer='glorot_uniform'
        )(window)
        window = tf.keras.layers.LeakyReLU()(window)
        window = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(3,3), strides = 1, kernel_initializer='glorot_uniform'
        )(window)
        window = tf.keras.layers.LeakyReLU()(window)
        window = tf.keras.layers.Flatten()(window)
        window = tf.keras.layers.Dense(units=32, kernel_initializer='glorot_uniform')(window) 
        window = tf.keras.layers.LeakyReLU()(window)        
        window = tf.keras.layers.Dense(units=32, kernel_initializer='glorot_uniform')(window) #tanh for segmented data only
        window = tf.keras.layers.LeakyReLU()(window)
        window = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform')(window)
        
        window = tf.keras.layers.LeakyReLU()(window)

        encoder = tf.keras.models.Model(inputs = window_inp, outputs = window)
        return encoder
    
    
    def v7(self):
        """intensity window input, and position input. One hot encoding. Do not change"""
        
        dims = (self.window_shape[0], self.window_shape[1], 2)
        window_inp = tf.keras.layers.Input(shape=dims)
        window = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(self.window_shape[0],self.window_shape[1]), strides = 1, kernel_initializer='glorot_uniform',use_bias=False,
        )(window_inp)
        window = tf.keras.layers.LeakyReLU()(window)
        window = tf.keras.layers.Dense(units=1, kernel_initializer='glorot_uniform')(window)        
        window = tf.keras.layers.ReLU()(window)

        encoder = tf.keras.models.Model(inputs = window_inp, outputs = window)
        return encoder


# class Torch_architectures(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=(3,3))
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3))
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3))
#         self.fc1 = nn.Linear(32*7*7, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.bn = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 1)
        

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x))
#         x = F.leaky_relu(self.conv2(x))
#         x = F.leaky_relu(self.conv3(x))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = self.bn(x)
#         x = F.leaky_relu(self.fc3(x))
#         return x

# %%
