import cv2 as cv 
import numpy as np
import os

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
img_name = input("Enter name:")
dir_ = r"C:\Users\samue\Computer_Vision\FaceData"
names = []
names.append(img_name)
for face in names:
    if not os.path.exists(face):
        path = os.path.join(dir_, face)
        os.mkdir(path)
    

for folder in names:
    id_ = 0
    user_input = input("Press s to start fetching faces: ").lower()
    if user_input != 's':
        print('Invalid input!')
        exit()
    while id_ <= 1000:
        _, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.10, 2)
        # for x, y, w, h in faces:
        #     cv.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 90), 5)
        #     roi = frame[y:y+h, x:x+w]
            
        cv.imshow("Feed", frame)
        
        
        filename = 'C:/Users/samue/Computer_Vision/FaceData/'+folder+'/img'+str(id_)+'.png'
        cv.imwrite(filename, gray)
        id_ += 1
       
        if cv.waitKey(1)&0xFF == ord('x'):
            break

print("faces Collected!")


cap.release()
cv.destroyAllWindows()