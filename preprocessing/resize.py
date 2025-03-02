import os
import constants
import numpy as np
from scipy import misc
from PIL import Image

def resize(image, dim1, dim2):
    # PIL을 사용해 이미지 크기 조정
    image = Image.fromarray(image)
    return image.resize((dim2, dim1))  # PIL에서 (width, height)로 크기 설정

def fileWalk(directory, destPath):
    try: 
        os.makedirs(destPath)
    except OSError:
        if not os.path.isdir(destPath):
            raise

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if len(file) <= 4 or file[-4:] != '.jpg':  # .jpg 확장자 필터
                continue

            # 이미지 로드 및 크기 조정
            pic = np.array(Image.open(os.path.join(subdir, file)))
            dim1 = len(pic)
            dim2 = len(pic[0])
            if dim1 > dim2:  # 세로 사진인 경우 회전
                pic = np.rot90(pic)

            picResized = resize(pic, constants.DIM1, constants.DIM2)
            picResized.save(os.path.join(destPath, file))  # 조정된 이미지를 저장

def main():
    # 원본 데이터 경로 설정
    prepath = r"data\dataset-original"
    glassDir = os.path.join(prepath, 'glass')
    paperDir = os.path.join(prepath, 'paper')
    cardboardDir = os.path.join(prepath, 'cardboard')
    plasticDir = os.path.join(prepath, 'plastic')
    metalDir = os.path.join(prepath, 'metal')
    trashDir = os.path.join(prepath, 'trash')
    styrofoamDir = os.path.join(prepath, 'styrofoam')  # 스티로폼 경로 추가

    # 전처리 후 데이터 저장 경로
    destPath = r"data\dataset-resized"
    try: 
        os.makedirs(destPath)
    except OSError:
        if not os.path.isdir(destPath):
            raise

    # 각 클래스별 데이터 전처리
    fileWalk(glassDir, os.path.join(destPath, 'glass'))
    fileWalk(paperDir, os.path.join(destPath, 'paper'))
    fileWalk(cardboardDir, os.path.join(destPath, 'cardboard'))
    fileWalk(plasticDir, os.path.join(destPath, 'plastic'))
    fileWalk(metalDir, os.path.join(destPath, 'metal'))
    fileWalk(trashDir, os.path.join(destPath, 'trash'))
    fileWalk(styrofoamDir, os.path.join(destPath, 'styrofoam'))  # 스티로폼 전처리

if __name__ == '__main__':
    main()
