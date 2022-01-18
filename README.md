# Attendance-System-based-on-Facial-recognition-Attendance-data-stored-in-a-csv-file-
Simple Python project using Opencv and datetime package to recognise faces and log attendance data in a csv file.
In "face_fetching.py" file I've automated the process of creating a custom dataset in an already existing folder. Each subfolder in the the "FaceData" folder has images  corresponding to the folder name. (Example: Samuel folder contains images of samuel)
The "Face_sample_train.py" file trains a model to recognize the faces in the dataset.
In the "Face_sample_train.py" file make sure you change the image directory accordingly with your very own. 
The trained model "trainz.yml" is implemented in the "Face_attend.py" program to recognize and log attendance data.
