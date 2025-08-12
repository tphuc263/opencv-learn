import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = './Resources/Faces/train'

haar_cascade = cv.CascadeClassifier('./haar_face.xml')
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person) # convert name (string) -> unique number (integer)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("train done")

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train recognizer on features list and labels list
features = np.array(features, dtype="object")
labels = np.array(labels)
face_recognizer.train(features, labels)

# save
script_dir = os.path.dirname(os.path.abspath(__file__))
face_recognizer.save(os.path.join(script_dir, 'face_trained.yml'))
np.save(os.path.join(script_dir, 'features.npy'), features)
np.save(os.path.join(script_dir, 'labels.npy'), labels)