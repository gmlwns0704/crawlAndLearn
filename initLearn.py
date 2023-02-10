# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4
@author: jongwon Kim
         Deep.I Inc.
"""

# 소스 원본 및 참고
# https://keras.io/ko/
# https://diane-space.tistory.com/178

import os
import shutil
from multiprocessing import Pool

import cv2
import urllib.request as urlreq
import json

import numpy as np
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
    # 사용자 값 입력
    categories = input('categories : ').split()
    keywordList = list() # 각 카테고리에 대한 키워드리스트
    keywordJson = dict()
    for strTmp in categories:
        tmp = list()
        for s in input('keywords of ' + strTmp + ' : ').split(): # 카테고리의 키워드 입력받음
            tmp.append(s.replace('_', ' ')) # _를 공백으로 교체
        keywordList.append(tmp)
        keywordJson[strTmp] = tmp

    searchTime = int(input('time to search keyword : '))

    keywordJson["categories"] = categories
    keywordJson["searchTime"] = searchTime
    with open('keywords.json', 'w', encoding='utf-8') as f:
        json.dump(keywordJson, f, indent='\t')


    # 디렉토리 초기화
    trainDir = 'train'
    testDir = 'test'
    print("start initializing '" + trainDir + "' directory...")
    if not os.path.exists(trainDir):
        print("'" + trainDir + "' directory is not found...")
        os.makedirs(trainDir)
        print("new '" + trainDir + "' directory is created..")
    else:  # 디렉토리 비우기
        print("'" + trainDir + "' directory is already existing, start emptying directory...")
        shutil.rmtree(trainDir)
        os.makedirs(trainDir)
        print("finished emptying directory...")
    print("finished initializing '" + trainDir + "' directory...")

    # 디렉토리 초기화
    testDir = 'test'
    print("start initializing '" + testDir + "' directory...")
    if not os.path.exists(testDir):
        print("'" + testDir + "' directory is not found...")
        os.makedirs(testDir)
        print("new '" + testDir + "' directory is created..")
    else:  # 디렉토리 비우기
        print("'" + testDir + "' directory is already existing, start emptying directory...")
        shutil.rmtree(testDir)
        os.makedirs(testDir)
        print("finished emptying directory...")
    print("finished initializing '" + testDir + "' directory...")

    # 크롬드라이버 불러오기
    # print('loading chrome driver...')
    # driver = webdriver.Chrome(ChromeDriverManager().install())
    # driver.implicitly_wait(3)

    # 키워드 선택
    # for i, category in enumerate(categories):
    #     f = open(category,'w')
    #     for keyword in keywordList[i]:
    #         searchAndDownload(category, keyword, driver, searchTime)

    # https://diane-space.tistory.com/178
    # 레이어 1
    model = Sequential()  # 모델 생성
    # model.add(Flatten(input_shape=(64, 64, 3)))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 레이어 2
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # 레이어3
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Fully Connected
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # 카테고리의 개수만큼 최종 출력뉴런
    model.add(Dense(len(categories), activation="softmax"))

    # 모델 실행 옵션
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    model.summary()  # 정보출력

    # 모델 저장
    model.save("model.h5")

    # trainDataGen = ImageDataGenerator(  # 학습데이터 생성
    #     rescale=1. / 255,  # 값을 1-0사이로 변경
    #     rotation_range=30,  # 무작위 회전각
    #     shear_range=0.2,  # 밀기
    #     zoom_range=0.4,  # 줌
    #     horizontal_flip=True  # 가끔 뒤집기
    # )
    # trainGenerator = trainDataGen.flow_from_directory(
    #     directory=trainDir,
    #     target_size=(128, 128),
    #     class_mode='categorical',
    #     batch_size=32
    # )
    # model.fit_generator(
    #     trainGenerator,
    #     epochs=10
    # )


if __name__ == "__main__":
    main()
