import cv2 as cv

# # read images
img = cv.imread('../Resources/Photos/cat.jpg')
cv.imshow('cat', img)
cv.waitKey(0)

# read videos
capture = cv.VideoCapture('../Resources/Videos/cat_video.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('cat see the sky', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()