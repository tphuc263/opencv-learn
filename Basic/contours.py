import cv2 as cv
import numpy as np
from Advanced import colors

img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
#
# blur = cv.GaussianBlur(gray, (5, 5), 0)
# cv.imshow('Blur', blur)

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny edges', canny)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f'{len(contours)} contours found!')

cv.drawContours(blank, contours, -1, colors.RED, thickness=1)
cv.imshow('Contours', blank)

cv.waitKey(0)