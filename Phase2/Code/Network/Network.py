"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

from multiprocessing import pool
import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    
    # prLogits, prSoftMax = my_NN(Img, ImageSize, MiniBatchSize)
    # prLogits, prSoftMax = ResNet(Img, ImageSize, MiniBatchSize)
    # prLogits, prSoftMax = ResNext(Img, ImageSize, MiniBatchSize)
    prLogits, prSoftMax = DenseNet(Img, ImageSize, MiniBatchSize)

    return prLogits, prSoftMax

def my_NN(Img, ImageSize, MiniBatchSize):
    x = Img
    x = tf.layers.conv2d(inputs = x, padding='same',filters = 32, kernel_size = 5, activation = None)
    x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True)
    x = tf.nn.relu(x)
    x  = tf.layers.max_pooling2d(inputs = x, pool_size = 2, strides = 2)

    x = tf.layers.conv2d(inputs = x, padding= 'same', filters = 64, kernel_size = 5, activation = None)
    x = tf.layers.batch_normalization(inputs = x,axis = -1, center = True, scale = True)
    x = tf.nn.relu(x)
    x  = tf.layers.max_pooling2d(inputs = x, pool_size = 2, strides = 2)

    x = tf.layers.flatten(x)

    #Define the Neural Network's fully connected layers:
    x = tf.layers.dense(inputs = x, units = 100, activation = tf.nn.relu)

    x = tf.layers.dense(inputs = x, units = 10, activation=None)

    prLogits = x
    prSoftMax = tf.nn.softmax(logits = prLogits)
    return prLogits, prSoftMax

def ResNet(Img, ImageSize, MiniBatchSize):
    '''
    x_origin = Input
    convolution layer (3x3)
    batch norm
    ReLU
    convolution layer (3x3)
    batch norm
    Add (Input + x_origin)
    ReLU
    '''
    x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.layers.batch_normalization(inputs=x)
    x_origin = x

    x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs=x, padding='same', filters = 16, kernel_size=3)
    x = tf.layers.batch_normalization(inputs=x)
    # downsample
    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)
    x_origin = tf.layers.max_pooling2d(inputs=x_origin, pool_size=2, strides=2)
    x = tf.math.add(x, x_origin)
    x = tf.nn.relu(x)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=10, activation=None)
    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)
    return prLogits, prSoftMax

def split(x, cardinality):
    out = []
    for _ in range(cardinality):
        x = tf.layers.conv2d(inputs=x, padding='same', filters = 16, kernel_size=3)
        x = tf.layers.batch_normalization(inputs=x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(inputs=x, padding='same', filters = 16, kernel_size=3)
        x = tf.layers.batch_normalization(inputs=x)
        out.append(x)
    x = tf.concat(out, axis=3)
    return x

def ResNext(Img, ImageSize, MiniBatchSize):
    x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.layers.batch_normalization(inputs=x)
    x_origin = x

    # x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    cardinality = 5
    x = split(x, cardinality)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.layers.batch_normalization(inputs=x)
    x = tf.math.add(x, x_origin)
    x = tf.nn.relu(x)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=10, activation=None)
    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax

def dense_block(input, repeat):
    for _ in range(repeat):
        x_input = input
        x = tf.layers.conv2d(input, padding='same', filters = 16, kernel_size=3)
        x = tf.layers.batch_normalization(x)
        x = tf.layers.separable_conv2d(x, padding='same', filters = 16, kernel_size=3)
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x)
        x = tf.concat([x_input,x],3)
        input = x
    return x

def DenseNet(Img, ImageSize, MiniBatchSize):
    x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.layers.batch_normalization(inputs=x)

    # x = tf.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    repeat = 4
    x = dense_block(x, repeat)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=10, activation=None)
    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax
