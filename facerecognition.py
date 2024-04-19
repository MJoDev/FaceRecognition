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

# Avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#Setup Paths

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

#Make the directories

os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

