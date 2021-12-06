import cv2 as cv
import time
import pickle
import os
import numpy as np
import openpyxl as op
import datetime as dt 
import time
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv.face.LBPHFaceRecognizer_create()
cap = cv.VideoCapture(0)

#img = cv.imread("sam.jpg")
recognizer.read("trainz.yml")
labels = {"person_name": 1}
with open("labels.pickles", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}


def markAttend(name):
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()
        name_list = []
        for line in data:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            date_ = dt.date.today()
            time_ = dt.datetime.today().time().strftime("%H:%M:%S")
            f.writelines(f'\n{labels[id_]},  {str(date_.strftime("%A-%B-%d-%Y"))}, {time_}')


while True:
    
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) 
    
    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.rectangle(frame, (x+w, y-48), (x, y),  (0, 255, 0), cv.FILLED)
        roi = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        #print(roi)
        id_, conf = recognizer.predict(roi_gray)
        if conf >=25 and conf <= 85:                                                                                                                                                      
            #print(id_)
            
            color = (155, 0, 200)
            # welc_ = f'Welcome {labels[id_]}'
            cv.putText(frame, labels[id_], (x+6, y-6), cv.FONT_HERSHEY_PLAIN, 2, color, 2)
            # textfile = open("Attendance.txt", "w")
            # for element in labels:
            #     textfile.write(str(element) + " " +str(dt.time()) + "\n")

            # textfile.close()
        else:
            color = (0, 0, 100)
            cv.putText(frame, "UnKnown", (x, y), cv.FONT_HERSHEY_PLAIN, 2, color,2)
    cv.imshow("feed", frame)  

  
    if cv.waitKey(1) & 0xFF == ord('x'):    
        break

markAttend(labels[id_])
cap.release()
cv.destroyAllWindows()    