import cv2 as cv
import numpy as np

img = cv.imread('../Photos/cat.jpg')
cv.imshow('Stupid Cat', img)

# translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
#  x --> Right
#  y --> Down

translated = translate(img, 100, 100)
cv.imshow('Translated', translated)

# rotation
def rotate(img, angle, rotPoint=None):
    (h, w) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (w//2, h//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale=1.0)
    dimensions = (w, h)
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)

rotated_rotated = rotate(rotated, -45)
cv.imshow('Rotated Rotated', rotated_rotated)

# resizing
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# flipping
flip = cv.flip(img, 1)
cv.imshow('Flipped', flip)

# cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)