import cv2  # Package of AI 
import numpy as np

#Lets capture the camera , 0 for webcam 
cap=cv2.VideoCapture(0)

# Lets Load the frame

while True:
    _,frame=cap.read()

    # we convert this format to hsv, bgr library this is color red , green and blue
    hsv_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Blue color 
    low_blue= np.array([94,80,2])
    high_blue=np.array([126,255,255])

    blue_mask=cv2.inRange(hsv_frame,low_blue,high_blue)
    blue= cv2.bitwise_and(frame,frame,mask=blue_mask)

#Lets frame on windows
    cv2.imshow("Frame", frame)
    cv2.imshow('Blue',blue)


# weight key event which is 1 and which is 27 then break the loop 

    key=cv2.waitKey(1)
    if key==27:
        break