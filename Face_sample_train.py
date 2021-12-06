import os
import numpy as np
import cv2 as cv
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
face_dir = os.path.join(base_dir, "FaceData")
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()

#print(face_dir)

# def resize(img, scale):
#     width = int(img.shape[1] * scale)
#     height = int(img.shape[0] * scale)
#     dimension = (width, height)
#     return cv.resize(img, dimension)
x_train = []
y_labels = []
label_id = {}
current_id = 0

for root, dirs, files in os.walk(face_dir):
     for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-")
            #print(label, path)
            fetch = cv.imread(path)
            gray = cv.cvtColor(fetch, cv.COLOR_BGR2GRAY)
            image_array = np.array(fetch, "uint8")
            #print(image_array)
            if not label in label_id: 
                label_id[label] = current_id
                current_id = current_id + 1

            id_ = label_id[label]
            print(label_id)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            

            for x, y, w, h in faces:
                roi = gray[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)



#print(y_labels)
#print(x_train)
with open("labels.pickles", "wb") as f:
    pickle.dump(label_id, f)


recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainz.yml")
