import cv2 as cv
import numpy as np
from Advanced import colors

# blank image
blank = np.zeros((500, 500, 3), dtype='uint8') # uint8 -> data type of an image

# # 1. paint the image
blank[200:300, 300:400] = colors.CYAN

# 2. draw a rectangle
cv.rectangle(blank, (0,0), (250, 500), colors.GREEN, thickness=cv.FILLED)

# 3. draw a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, colors.RED, thickness=3)

# 4.draw a line
cv.line(blank, (100, 250), (blank.shape[1]//2, blank.shape[0]//2), colors.GRAY, thickness=3)

# 5. write text
cv.putText(blank, "Hello, broken interview process", (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, colors.GREEN, thickness= 2)

cv.imshow('frame', blank)
cv.waitKey(0)
