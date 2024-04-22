#import standard dependencies
import cv2
import os
import random 
import numpy as np
from matplotlib import pyplot as plt

#import tensorflow dependecies - Funtional API
from tensorflow import keras
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.models import Model 
import tensorflow as tf

#import uuid library to generate unique image names
import uuid


# Avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Setup Paths

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


#Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #Establish 250
    frame = frame[80:80+250, 200:200+250, :]

    #Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        #Creating a unique file name
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(imgname, frame)
    #Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        #Creating a unique file name
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(imgname, frame)
    #Show image back to screen
    cv2.imshow('Image Collection', frame)

    #Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

anchor = tf.data.Dataset.list_files(ANC_PATH+'\.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH+'\.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH+'\.jpg').take(300)

def preprocess(file_path):
    #Reading image for file path
    byte_img = tf.io.read_file(file_path)
    # Load the image
    img = tf.io.decode_jpeg(byte_img)

    #Processing steps - resizing image
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

data = positives.concatenate(negatives)

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)
