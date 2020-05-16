# Import Lib
import matplotlib.pyplot as plt
import cv2
import numpy as np

def convert(img, color_space='bgr2hsv'):
    return cv2.cvtColor(img, eval('cv2.COLOR_%s' % color_space.upper()))

def thresholding(img):
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def drawContours(img, contours):
    _ = cv2.drawContours(img, contours, -1, 255, -1)
    return img

def floodFill(img):
    ff = img.copy()
    contours, _ = cv2.findContours(ff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _ = cv2.drawContours(ff, contours, -1, 255, -1)
    return ff


def erosion(img):
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erode = cv2.erode(img, kernel, iterations = 3)
    return erode

def main(path):
    image = cv2.imread(path)
    image_YCrCb = convert(image, 'BGR2YCr_Cb')
    image_Cb = image_YCrCb[...,2]
    # ~ thresholding to take coin value
    image_thresholding = ~thresholding(image_Cb) 
    image_floodFill = floodFill(image_thresholding)
    image_Erosion = erosion(image_floodFill)
    image_contours_2 = contours(image_Erosion)
    img_Final = image.copy()
    for cnt in image_contours_2:
        epsilon = 0.01* cv2.arcLength(cnt, True)
        new_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        dc_new = cv2.drawContours(img_Final, cnt, -1, (0,255,0), 2)
    putText = cv2.putText(img_Final, "Number of coins: " + str(len(image_contours_2)), (40,40), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 255, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False)
    print("Image at D:\Coins_detection\Result")
    return img_Final
    

def saveImg(img, path):
    cv2.imwrite(path, img)
if __name__ == "__main__":
    path = 'Coins - 4.jpg'
    image = main(path)
    saveImg(image, './Result/%s' %path.split("/")[-1])






