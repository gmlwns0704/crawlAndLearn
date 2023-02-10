import json
import os
import time
from multiprocessing import Pool
from urllib import request

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


def searchAndDownload(category, keyword, searchTime=30, driver=None, trainDir='train', testDir='test'):
    if driver is None:
        # 크롬드라이버 불러오기
        print('loading chrome driver...')
        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.implicitly_wait(3)
    # 검색
    print('start searching keyword : ' + keyword + '...')
    driver.get('https://www.google.co.kr/imghp?hl=ko')  # 구글이미지검색(kor)
    search = driver.find_element(by=By.NAME, value='q')
    search.send_keys(keyword)  # 키워드 전달
    search.send_keys(Keys.RETURN)  # 엔터

    # 브라우저 스크롤
    print('scrolling browser...')
    body = driver.find_element(by=By.TAG_NAME, value='body')  # 브라우저의 html body

    t = time.time()
    while time.time() - t < searchTime:
        for i in range(60):
            if time.time() - t >= searchTime:
                break
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.1)
        try:
            driver.find_element(by=By.XPATH, value='//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input').click()
        except:
            pass

    # 이미지들의 목록을 images로 받음
    images = driver.find_elements(By.CSS_SELECTOR, value='img.rg_i.Q4LuWd')  # 썸네일 이미지의 class, 해당 class의 모든 데이터 읽어옴
    print('number of image url with keyword ' + keyword + ' :', len(images), ', start reading urls...')

    # 이미지들의 url과 번호를 links에 저장
    links = []  # 이미지들의 url
    for image in tqdm(images):
        if image.get_attribute('src') is not None:
            # 저장구조 : '(번호) (url)'
            links.append(image.get_attribute('src'))  # links에 추가

    print('number of available image url with keyword ' + keyword + ' :', str(len(links)), ', start reading urls...')

    # 해당 키워드의 디렉터리 생성
    if not os.path.exists(trainDir + '/' + category):
        os.makedirs(trainDir + '/' + category)

    if not os.path.exists(testDir + '/' + category):
        os.makedirs(testDir + '/' + category)

    # 검색한 이미지 지정된 사이즈로 변환해서 저장
    for i, link in enumerate(tqdm(links)):
        req = request.urlopen(link)
        imageNparray = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(imageNparray, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
        if i <= len(links) * 0.9: # 90퍼센트는 학습용
            cv2.imwrite(trainDir + '/' + category + '/' + keyword + str(i) + '.png', img)
        else: # 10퍼센트는 테스트용
            cv2.imwrite(testDir + '/' + category + '/' + keyword + str(i) + '.png', img)

    # 최종 결과 출력``
    print('total found urls : ' + str(len(images)))
    print('total available urls : ' + str(len(links)))
    print('total saved keyword images : ' + str(len(os.listdir(trainDir+'/'+category))))

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

    # 키워드 선택 및 다운로드
    for i, category in enumerate(categories):
        arg = list()
        for keyword in keywordList[i]:
            arg.append((category, keyword, searchTime))
        print(arg)
        procNum = min(5, len(keywordList[i]))
        with Pool(processes=procNum) as p:
            p.starmap(searchAndDownload, arg)

if __name__ == "__main__":
    main()