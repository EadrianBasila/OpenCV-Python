import cv2
import numpy as np
import pytesseract
import os



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('OpenCV\Query.png')
h,w,c = imgQ.shape
imgQ = cv2.resize(imgQ, (w//3,h//3))  

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
impKp1 = cv2.drawKeypoints(imgQ, kp1, None)

cv2.imshow('Keypoints', kp1)
cv2.imshow('Output', imgQ)
cv2.waitKey(0)