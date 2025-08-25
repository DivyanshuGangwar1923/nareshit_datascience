import numpy as np
import cv2

#Load the haar cascade for fave detection

face_classifier=cv2.CascadeClassifier(r"C:\Users\divya\Downloads\haarcascade_frontalface_default.xml")

#Load the image

image=cv2.imread(r"C:\Users\divya\Downloads\IMG_4911_320250813171633222.jpg")

#Convert the image to grey scale 

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Detect faces in the image
faces=face_classifier.detectMultiScale(gray,1.3,5)

if len(faces)==0:
    print('no face found!')
else:
    #Draw rectangles arounf the faces
    for(x,y,w,h) in faces: #(x,y) is the top-left corner and w,h is thw width
        cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey(0)

cv2.destroyAllWindows()