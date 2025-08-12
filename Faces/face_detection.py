import cv2 as cv


img = cv.imread('../Resources/Photos/group 2.jpg')
cv.imshow('Group persons', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

haar_cascade = cv.CascadeClassifier('../haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
print("number of faces detected: ", len(faces_rect))

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow('Faces detected', img)


cv.waitKey(0)