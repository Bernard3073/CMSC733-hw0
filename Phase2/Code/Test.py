#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import CIFAR10Model
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
# from StringIO import StringIO
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import seaborn as sns

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.png'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.png')

    return ImageSize, DataPath
    
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    # I1S = iu.StandardizeInputs(np.float32(I1))
    I1S = np.float32(I1) / 255
    I1S -= 0.5
    I1S *= 2

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1
                

def TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred, NetworkType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    Length = ImageSize[0]
    # Predict output with forward pass, MiniBatchSize for Test is 1
    _, prSoftMaxS = CIFAR10Model(ImgPH, ImageSize, 1, NetworkType)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        OutSaveT = open(LabelsPathPred, 'w')

        for count in tqdm(range(np.size(DataPath))):            
            DataPathNow = DataPath[count]
            Img, ImgOrg = ReadImages(ImageSize, DataPathNow)
            FeedDict = {ImgPH: Img}
            PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))

            OutSaveT.write(str(PredT)+'\n')
            
        OutSaveT.close()

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsTrue, LabelsPred)), '%')
    Txtfile = open("Test_graph/Accuracy.txt", "a")
    Txtfile.write('Accuracy: '+ str(Accuracy(LabelsTrue, LabelsPred)) + '%')
    Txtfile.close()
    return cm

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/bernard/CMSC733/hw0/Phase2/Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/bernard/CMSC733/hw0/Phase2/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--NetworkType', default='my_NN',
                        help='Path to save Logs for Tensorboard, Default=my_NN')
    Args = Parser.parse_args()
    # ModelPath = Args.ModelPath + Args.NetworkType + '/'+'49model.ckpt'
    # BasePath = Args.BasePath
    # LabelsPath = Args.LabelsPath
    # NetworkType = Args.NetworkType
    

    # # Setup all needed parameters including file reading
    # ImageSize, DataPath = SetupAll(BasePath)

    # # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    # LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    # # TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred, NetworkType)

    # # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # cm = ConfusionMatrix(LabelsTrue, LabelsPred)
    # df_cm = pd.DataFrame(cm, range(10),range(10))
    # #plt.figure(figsize = (10,7))
    # sns.set(font_scale=1.4)#for label size
    # sns.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    # plt.show()
    accTestOverEpochs=np.array([0,0])
    for epoch in tqdm(range(50)):
        # Parse Command Line arguments
        tf.reset_default_graph()

        ModelPath = Args.ModelPath + Args.NetworkType + '/' + str(epoch)+'model.ckpt'
        # print(ModelPath)
        BasePath = Args.BasePath
        LabelsPath = Args.LabelsPath

        # Setup all needed parameters including file reading
        ImageSize, DataPath = SetupAll(BasePath)

        # Define PlaceHolder variables for Input and Predicted output
        ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
        LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

        TestOperation(ImgPH, ImageSize, ModelPath, DataPath, LabelsPathPred, Args.NetworkType)

        # Plot Confusion Matrix
        LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
        # for Python 3: change 'map' to 'list'
        LabelsTrue = list(LabelsTrue)
        LabelsPred = list(LabelsPred)
        accuracy=Accuracy(LabelsTrue, LabelsPred)
        accTestOverEpochs=np.vstack((accTestOverEpochs,[epoch,accuracy]))
    plt.xlim(0,60)
    plt.ylim(0,100)
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy (%)')
    plt.subplots_adjust(hspace=0.6,wspace=0.3)
    plt.plot(accTestOverEpochs[:,0],accTestOverEpochs[:,1])
    plt.savefig('Test_graph/'+Args.NetworkType+'/Epoch_acc.png')
    plt.close()

    cm = ConfusionMatrix(LabelsTrue, LabelsPred)
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('Test_graph/'+Args.NetworkType+'/confusion_matrix.png')
    plt.close() 

if __name__ == '__main__':
    main()
 
