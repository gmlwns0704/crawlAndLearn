import os
import shutil

import cv2
import urllib.request as urlreq

import keras
import numpy as np
from keras.applications import VGG16
from keras.applications.convnext import decode_predictions
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time


def main():
    model = keras.models.load_model('model.h5')
    # model = VGG16(weights='imagenet', include_top=True)

    testDir = 'test'

    testDataGen = ImageDataGenerator(rescale=1./255)
    testGenerator = testDataGen.flow_from_directory(
        directory=testDir,
        target_size=(64, 64),
        batch_size=32
    )

    #테스트
    loss, accuracy = model.evaluate_generator(
        testGenerator
    )

    print('loss : '+str(loss))
    print('acc : '+str(accuracy))


if __name__ == "__main__":
    main()