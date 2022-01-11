import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# def dl_version(input, iterate=500):

a  = cv.imread('./lena.jpg', cv.IMREAD_GRAYSCALE)
a = cv.resize(a, (0, 0), fx=0.2, fy=0.2)
print(a.shape)
hist=cv.calcHist([a],[0],None,[256],[0,256])
plt.hist(a.ravel(),256,[0,256])
plt.show()
b = cv.imshow('hey', a)
cv.waitKey()