import cv2 as cv
import numpy as np

img = cv.imread('../Photos/park.jpg')
cv.imshow('Park', img)

# convert BGR (multiple channel) -> 1 channel, from 0 black to 255 white
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian -> dao ham bac 1
lap = cv.Laplacian(gray, cv.CV_64F) # -> make output float -> can store negative value
lap = np.uint8(np.absolute(lap)) # abs -> display image in range 0 - 255
cv.imshow('Laplacian', lap)

# Sobel -> dao ham bac 2 (gradient) -> caculate gradient on different dimension x and y
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0) # gradient bậc 1 trục x, bậc 0 trục y -> nổi biên dọc
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1) # gradient bậc 0 trục x, bậc 1 trục y -> nổi biên ngang
combined_sobel = cv.bitwise_or(sobelx, sobely) # --> combine result

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

# Canny -> gold standard -> 4 step
canny = cv.Canny(gray, 150, 175) # gradient > 175 -> strong edge, < 150 -> remove, >=150 and <= 175 -> keep if by side of an strong edge
cv.imshow('Canny', canny)
cv.waitKey(0)