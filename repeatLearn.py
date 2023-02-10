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
import json
import keras
import matplotlib.pyplot as plt

from multiprocessing import Pool
from keras.preprocessing.image import ImageDataGenerator

from searchAndDownload import searchAndDownload


def main():
    # json에서 데이터 읽어옴
    print('start reading json...')
    with open('keywords.json', 'r') as f:
        keywordJson = json.load(f)
    print(json.dumps(keywordJson, indent='\t'))

    categories = keywordJson['categories']
    keywordList = list()
    for s in categories:
        keywordList.append(keywordJson[s])

    searchTime = keywordJson['searchTime']
    beforeHist = dict()
    if 'history' in keywordJson:
        beforeHist = keywordJson['history']
    epochs = 10


    # 디렉토리 없으면 재생성
    trainDir = 'train'
    testDir = 'test'
    print("start initializing '" + trainDir + "' directory...")
    if not os.path.exists(trainDir):
        print("'" + trainDir + "' directory is not found...")
        os.makedirs(trainDir)
        print("new '" + trainDir + "' directory is created..")

    testDir = 'test'
    print("start initializing '" + testDir + "' directory...")
    if not os.path.exists(testDir):
        print("'" + testDir + "' directory is not found...")
        os.makedirs(testDir)
        print("new '" + testDir + "' directory is created..")

    # 성능개선이 멈출때까지 무한반복
    while True:
        # 모델 불러오기
        model = keras.models.load_model('model.h5')
        # 학습데이터 생성
        trainDataGen = ImageDataGenerator(  # 학습데이터 생성
            rescale=1. / 255,  # 값을 1-0사이로 변경
            rotation_range=30,  # 무작위 회전각
            shear_range=0.2,  # 밀기
            zoom_range=0.4,  # 줌
            horizontal_flip=True  # 가끔 뒤집기
        )
        trainGenerator = trainDataGen.flow_from_directory(
            directory=trainDir,
            target_size=(64, 64),
            class_mode='categorical',
            batch_size=32
        )
        # 학습
        hist = model.fit_generator(
            trainGenerator,
            epochs=epochs
        )
        # https://sevillabk.github.io/1-early-stopping/
        # 모델 학습과정 표시하기
        # fig, loss_ax = plt.subplots()
        # acc_ax = loss_ax.twinx()
        #
        # loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        # acc_ax.plot(hist.history['accuracy'], 'b', label='train accuracy')
        #
        # loss_ax.set_xlabel('epoch')
        # loss_ax.set_ylabel('loss')
        # acc_ax.set_xlabel('accuracy')
        #
        # loss_ax.legend(loc='upper left')
        # acc_ax.legend(loc='lower left')
        #
        # plt.show()
        # 비교용 이전 히스토리 불러오기
        if 'history' in keywordJson:
            beforeHist = keywordJson['history']
        # 다음 루프여부 판단 - 이번 모델의 성능이 더 구리다면 루프 멈추고 저장도 안함
        if 'accuracy' in beforeHist and max(hist.history['accuracy']) < max(beforeHist['accuracy']):
            print('found best model, stop learning...')
            print('best accuracy : ' + str(max(beforeHist['accuracy'])))
            break
        if 'accuracy' in beforeHist:
            print('before accuracy : ' + str(beforeHist['accuracy']))
            print('current accuracy : ' + str(hist.history['accuracy']))

        print('accuracy improved, keep learning...')
        # 모델 저장
        model.save("model.h5")
        print('model saved')
        # 히스토리 저장
        keywordJson['history'] = hist.history
        with open('keywords.json', 'w') as f:
            json.dump(keywordJson, f, indent='\t')


if __name__ == "__main__":
    main()
