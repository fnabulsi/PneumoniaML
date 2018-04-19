import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob2
import time

def get_num_pixels(filepath):
    '''Gets the number of pixels of a picture

        Args:
                filepath: The full path of the image on the local machine
        Returns:
                size: The width*height of the picture
    '''
    width, height = Image.open(filepath).size
    return width*height

def processData(dataFilePath):
    '''Processes images in the data directory by collecting them in the "dataSet"
       data structure. This function shrinks the images from 1200x1200 to 120x120.

       Args:
            dataFilePath: The path of your data file including data itself
                          ex) /Users/ftnabulsi/cs596/data/
       Returns:
            dataSet: a map containing training data/labels and validation data/labels
    '''
    startTime = time.time()
    TRAINING_NORMAL = 'train/NORMAL/*.jpeg'
    TRAINING_PNEUMONIA = 'train/PNEUMONIA/*.jpeg'
    VALIDATION_NORMAL = 'val/NORMAL/*.jpeg'
    VALIDATION_PNEUMONIA = 'val/PNEUMONIA/*.jpeg'
    trainingData = []
    trainingLabels = []
    i = 1
    #Collect training images of negative X-rays
    for filename in glob2.glob(dataFilePath + TRAINING_NORMAL):
        if(i % 100 == 0):
            print('Processing image: ', i)
        image = cv.imread(filename)
        image = cv.resize(image, (0,0), fx=0.1, fy=0.1)
        trainingData.append(image)
        trainingLabels.append(0)
        i = i + 1
    i = 1
    #Collect training images of positive X-rays
    for filename in glob2.glob(dataFilePath + TRAINING_PNEUMONIA):
        if(i % 100 == 0):
            print('Processing image: ', i)
        image = cv.imread(filename)
        image = cv.resize(image, (0,0), fx=0.1, fy=0.1)
        trainingData.append(image)
        trainingLabels.append(1)
        i = i + 1
    print('Total number of training samples: ', len(trainingData))

    validationData = []
    validationLabels = []
    i = 1
    #Collect validation images for negative X-rays
    for filename in glob2.glob(dataFilePath + VALIDATION_NORMAL):
        if(i % 100 == 0):
            print('Processing image: ', i)
        image = cv.imread(filename)
        image = cv.resize(image, (0,0), fx=0.1, fy=0.1)
        validationData.append(image)
        validationLabels.append(0)
        i = i + 1

    i = 1
    #Collect validation images for positive X-rays
    for filename in glob2.glob(dataFilePath + VALIDATION_PNEUMONIA):
        if(i % 100 == 0):
            print('Processing image: ', i)
        image = cv.imread(filename)
        image = cv.resize(image, (0,0), fx=0.1, fy=0.1)
        validationData.append(image)
        validationLabels.append(1)
        i = i + 1
    
    dataSet = {
        'trainingImages': trainingData,
        'trainingLabels': trainingLabels,
        'validationImages': validationData,
        'validationLabels': validationLabels}
    endTime = time.time()
    trainingTime = endTime - startTime
    print("Total image processing time: ", trainingTime, "s")
    return dataSet
    
# plt.imshow(image, cmap='gray', interpolation='bicubic')
# plt.show()