import cv2 as cv

img = cv.imread('../Resources/Photos/cat.jpg')
cv.imshow('cat', img)

def rescaleFrame(frame, scale=0.75):
    # Images, videos, live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # live video
    capture.set(3, width)
    capture.set(4, height)

capture = cv.VideoCapture('../Resources/Videos/cat_video.mp4')
while True:
    isTrue, frame = capture.read()
    resize_frame = rescaleFrame(frame, scale=0.4)

    cv.imshow('cat see the sky', frame)
    cv.imshow('cat see the sky resired', resize_frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()