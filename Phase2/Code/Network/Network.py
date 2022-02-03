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

def CIFAR10Model(Img, ImageSize, MiniBatchSize, NetworkType):
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
    if NetworkType == 'my_NN':
        prLogits, prSoftMax = my_NN(Img, ImageSize, MiniBatchSize)
    elif NetworkType == 'ResNet':
        prLogits, prSoftMax = ResNet(Img, ImageSize, MiniBatchSize)
    elif NetworkType == 'ResNeXt':
        prLogits, prSoftMax = ResNext(Img, ImageSize, MiniBatchSize)
    elif NetworkType == 'DenseNet':
        prLogits, prSoftMax = DenseNet(Img, ImageSize, MiniBatchSize)

    return prLogits, prSoftMax

def my_NN(Img, ImageSize, MiniBatchSize):
    x = Img
    x = tf.compat.v1.layers.conv2d(inputs = x, padding='same',filters = 32, kernel_size = 5, activation = None)
    x = tf.compat.v1.layers.batch_normalization(inputs = x)
    x = tf.nn.relu(x)
    x  = tf.compat.v1.layers.max_pooling2d(inputs = x, pool_size = 2, strides = 2)

    x = tf.compat.v1.layers.conv2d(inputs = x, padding= 'same', filters = 64, kernel_size = 5, activation = None)
    x = tf.compat.v1.layers.batch_normalization(inputs = x)
    x = tf.nn.relu(x)
    x  = tf.compat.v1.layers.max_pooling2d(inputs = x, pool_size = 2, strides = 2)

    x = tf.compat.v1.layers.flatten(x)

    #Define the Neural Network's fully connected layers:
    x = tf.compat.v1.layers.dense(inputs = x, units = 100, activation = tf.nn.relu)

    x = tf.compat.v1.layers.dense(inputs = x, units = 10, activation=None)

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
    def residual_block(input, filters, kernel_size):
        input = tf.compat.v1.layers.conv2d(inputs=input, padding='same', filters = filters, kernel_size=kernel_size)
        input = tf.compat.v1.layers.batch_normalization(inputs=input)
        x_origin = input
        x = tf.compat.v1.layers.conv2d(inputs=input, padding='same', filters = filters, kernel_size=kernel_size)
        x = tf.compat.v1.layers.batch_normalization(inputs=x)
        x = tf.nn.relu(x)
        x = tf.compat.v1.layers.conv2d(inputs=x, padding='same', filters = filters, kernel_size=kernel_size)
        x = tf.compat.v1.layers.batch_normalization(inputs=x)
        # downsample
        x = tf.compat.v1.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)
        x_origin = tf.compat.v1.layers.max_pooling2d(inputs=x_origin, pool_size=2, strides=2)
        x = tf.math.add(x, x_origin)
        x = tf.nn.relu(x)
        return x
    def num_residual_block(x, num, filters, kernel_size):
        for _ in range(num):
            x = residual_block(x, filters, kernel_size)
        return x
    
    x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.compat.v1.layers.batch_normalization(inputs=x)
    x = num_residual_block(Img, 2, 16, 3)
    x = num_residual_block(Img, 2, 64, 3)

    x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='same')
    x = tf.compat.v1.layers.flatten(x)
    # x = tf.compat.v1.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    # x = tf.compat.v1.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    x = tf.compat.v1.layers.dense(inputs=x, units=10, activation=None)
    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)
    return prLogits, prSoftMax


def ResNext(Img, ImageSize, MiniBatchSize):
    def split_layer(x, cardinality, filters, kernel_size):
        out = []
        for _ in range(cardinality):
            x = tf.compat.v1.layers.conv2d(inputs=x, padding='same', filters = filters, kernel_size=kernel_size)
            x = tf.compat.v1.layers.batch_normalization(inputs=x)
            x = tf.nn.relu(x)
            x = tf.compat.v1.layers.conv2d(inputs=x, padding='same', filters = 2*filters, kernel_size=kernel_size)
            x = tf.compat.v1.layers.batch_normalization(inputs=x)
            out.append(x)
        x = tf.concat(out, axis=3)
        return x
    def transition_layer(x, filters, kernel_size):
        x = tf.compat.v1.layers.conv2d(inputs=x, padding='same', filters = filters, kernel_size=kernel_size)
        x = tf.compat.v1.layers.batch_normalization(inputs=x)
        x = tf.nn.relu(x)
        return x

    def residual_layer(input, res_block, cardinality, filters, kernel_size):
        x = tf.compat.v1.layers.conv2d(inputs=input, padding='same', filters = filters, kernel_size=kernel_size)
        x = tf.compat.v1.layers.batch_normalization(inputs=x)
        x_origin = x
        for _ in range(res_block):
            x = split_layer(input, cardinality, filters, kernel_size)
            x = transition_layer(x, filters, kernel_size)
            x = tf.math.add(x, x_origin)
        return x

    x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    x = tf.compat.v1.layers.batch_normalization(inputs=x)

    cardinality = 5
    # number of split and transition
    res_block = 1
    x = residual_layer(x, res_block, cardinality, 16, 3)
    x = residual_layer(x, res_block, cardinality, 64, 3)

    x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='same')
    x = tf.compat.v1.layers.flatten(x)
    x = tf.compat.v1.layers.dense(inputs=x, units=10, activation=None)
    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax

# def ResNext(Img, ImageSize, MiniBatchSize):
#     def split(x, cardinality):
#         out = []
#         for _ in range(cardinality):
#             x = tf.compat.v1.layers.conv2d(inputs=x, padding='same', filters = 16, kernel_size=1)
#             x = tf.compat.v1.layers.batch_normalization(inputs=x)
#             x = tf.nn.relu(x)
#             x = tf.compat.v1.layers.conv2d(inputs=x, padding='same', filters = 32, kernel_size=5)
#             x = tf.compat.v1.layers.batch_normalization(inputs=x)
#             out.append(x)
#         x = tf.concat(out, axis=3)
#         return x
#     x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
#     x = tf.compat.v1.layers.batch_normalization(inputs=x)
#     x_origin = x

#     # x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
#     cardinality = 5
#     x = split(x, cardinality)
#     x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
#     x = tf.compat.v1.layers.batch_normalization(inputs=x)
#     x = tf.math.add(x, x_origin)
#     x = tf.nn.relu(x)

#     x = tf.compat.v1.layers.flatten(x)
#     # x = tf.compat.v1.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
#     # x = tf.compat.v1.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
#     x = tf.compat.v1.layers.dense(inputs=x, units=10, activation=None)
#     prLogits = x
#     prSoftMax = tf.nn.softmax(logits=prLogits)

#     return prLogits, prSoftMax


def DenseNet(Img, ImageSize, MiniBatchSize):
    '''
    input
    conv
    dense block 1
    conv + pooling
    dense block 2
    conv + pooling
    dense block 3
    pooling
    linear
    output
    '''
    def bn_rl_conv(x, filters, kernel):
        # batch norm + relu + conv
        x = tf.compat.v1.layers.batch_normalization(inputs=x)
        x = tf.nn.relu(x)
        x = tf.compat.v1.layers.conv2d(x, padding='same', filters = filters, kernel_size=kernel)
        x = tf.layers.dropout(inputs=x, rate=0.2)
        return x

    def dense_block(x, repeat):
        for _ in range(repeat):
            # x_input = input
            # x = tf.compat.v1.layers.conv2d(input, padding='same', filters = 16, kernel_size=3)
            # x = tf.compat.v1.layers.batch_normalization(x)
            # x = tf.compat.v1.layers.separable_conv2d(x, padding='same', filters = 16, kernel_size=3)
            # x = tf.nn.relu(x)
            # x = tf.compat.v1.layers.batch_normalization(x)
            # x = tf.concat([x_input,x],3)
            # input = x
            y = bn_rl_conv(x, 64, 3)
            y = bn_rl_conv(y, 16, 3)
            x = tf.concat([y, x], axis=3)
        #     out.append(x)
        # x = tf.concat(out, axis=3)
        return x
    # def concatenation(nodes):
    #     return tf.concat(nodes,axis=3)
    # def dense_block(Img, len_dense):
    #     with tf.variable_scope("dense_unit"+str(1)):
    #         nodes = []
    #         # img = tf.layers.conv2d(inputs = Img,padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
    #         # img = tf.layers.conv2d(inputs = Img, padding='same',filters = 16, kernel_size = 5, activation = None)
    #         img = bn_rl_conv(Img, 64, 1, 1)
    #         img = bn_rl_conv(img, 16, 3, 1)
    #         nodes.append(img)
    #         for z in range(len_dense):
    #             img = tf.nn.relu(img)
    #             img = tf.layers.conv2d(inputs = img, padding='same',filters = 16, kernel_size = 5, activation = None)
    #             net = tf.layers.conv2d(inputs = concatenation(nodes), padding='same',filters = 16, kernel_size = 5, activation = None)
    #             nodes.append(net)
    #         return net

    def transition_layer(x):
        x = bn_rl_conv(x, x.shape[-1]//2, 1)
        x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='same')
        return x

    x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 32, kernel_size=7, strides=2)
    # x = tf.compat.v1.layers.batch_normalization(inputs=x)
    # x  = tf.compat.v1.layers.max_pooling2d(inputs = x, padding='same', pool_size = 3, strides = 2)

    # x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
    # repeat = 4
    # x = dense_block(x, repeat)
    # for repeat in [6, 12, 24, 16]:
    #     d = dense_block(x, repeat)
    #     x = transition_layer(d)

    x = dense_block(x, 6)
    x = transition_layer(x)

    x = dense_block(x, 12)
    x = transition_layer(x)

    x = dense_block(x, 24)
    x = transition_layer(x)

    x = dense_block(x, 16)
    x = tf.compat.v1.layers.batch_normalization(inputs=x)
    x = tf.nn.relu(x)

    x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='same')
    
    x = tf.compat.v1.layers.flatten(x) 
    x = tf.compat.v1.layers.dense(inputs=x, units=10, activation=None)
    prLogits = x
    prSoftMax = tf.nn.softmax(logits=prLogits)

    return prLogits, prSoftMax
# def DenseNet(Img, ImageSize, MiniBatchSize):
#     '''
#     input
#     conv
#     dense block 1
#     conv + pooling
#     dense block 2
#     conv + pooling
#     dense block 3
#     pooling
#     linear
#     output
#     '''
#     def bn_rl_conv(x, filters, kernel):
#         # batch norm + relu + conv
#         x = tf.compat.v1.layers.batch_normalization(inputs=x)
#         x = tf.nn.relu(x)
#         x = tf.compat.v1.layers.conv2d(x, padding='same', filters = filters, kernel_size=kernel)
#         x = tf.layers.dropout(inputs=x, rate=0.2)
#         return x

#     def dense_block(x, repeat):
#         for _ in range(repeat):
#             # x_input = input
#             # x = tf.compat.v1.layers.conv2d(input, padding='same', filters = 16, kernel_size=3)
#             # x = tf.compat.v1.layers.batch_normalization(x)
#             # x = tf.compat.v1.layers.separable_conv2d(x, padding='same', filters = 16, kernel_size=3)
#             # x = tf.nn.relu(x)
#             # x = tf.compat.v1.layers.batch_normalization(x)
#             # x = tf.concat([x_input,x],3)
#             # input = x
#             y = bn_rl_conv(x, 64, 3)
#             y = bn_rl_conv(y, 16, 3)
#             x = tf.concat([y, x], axis=3)
#         #     out.append(x)
#         # x = tf.concat(out, axis=3)
#         return x
#     # def concatenation(nodes):
#     #     return tf.concat(nodes,axis=3)
#     # def dense_block(Img, len_dense):
#     #     with tf.variable_scope("dense_unit"+str(1)):
#     #         nodes = []
#     #         # img = tf.layers.conv2d(inputs = Img,padding = 'same', filters = num_filters, kernel_size = kernel_size, activation = None)
#     #         # img = tf.layers.conv2d(inputs = Img, padding='same',filters = 16, kernel_size = 5, activation = None)
#     #         img = bn_rl_conv(Img, 64, 1, 1)
#     #         img = bn_rl_conv(img, 16, 3, 1)
#     #         nodes.append(img)
#     #         for z in range(len_dense):
#     #             img = tf.nn.relu(img)
#     #             img = tf.layers.conv2d(inputs = img, padding='same',filters = 16, kernel_size = 5, activation = None)
#     #             net = tf.layers.conv2d(inputs = concatenation(nodes), padding='same',filters = 16, kernel_size = 5, activation = None)
#     #             nodes.append(net)
#     #         return net

#     def transition_layer(x):
#         x = bn_rl_conv(x, x.shape[-1]//2, 1)
#         x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='same')
#         return x

#     x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=5, strides=2)
#     # x = tf.compat.v1.layers.batch_normalization(inputs=x)
#     # x  = tf.compat.v1.layers.max_pooling2d(inputs = x, padding='same', pool_size = 3, strides = 2)

#     # x = tf.compat.v1.layers.conv2d(inputs=Img, padding='same', filters = 16, kernel_size=3)
#     # repeat = 4
#     # x = dense_block(x, repeat)
#     # for repeat in [6, 12, 24, 16]:
#     #     d = dense_block(x, repeat)
#     #     x = transition_layer(d)

#     x = dense_block(x, 6)
#     x = transition_layer(x)

#     x = dense_block(x, 12)
#     x = transition_layer(x)

#     x = dense_block(x, 24)
#     x = transition_layer(x)

#     x = dense_block(x, 16)
#     x = tf.compat.v1.layers.batch_normalization(inputs=x)
#     x = tf.nn.relu(x)

#     x = tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='same')
    
#     x = tf.compat.v1.layers.flatten(x) 
#     x = tf.compat.v1.layers.dense(inputs=x, units=10, activation=None)
#     prLogits = x
#     prSoftMax = tf.nn.softmax(logits=prLogits)

#     return prLogits, prSoftMax
