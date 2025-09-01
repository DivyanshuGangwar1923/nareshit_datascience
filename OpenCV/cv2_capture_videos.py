import cv2  # Package of AI 
import numpy as np

#Lets capture the camera , 0 for webcam 
cap=cv2.VideoCapture(0)

# Lets Load the frame

while True:
    _,frame=cap.read()

    # we convert this format to hsv, bgr library this is color red , green and blue
    hsv_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Red color 
    low_red= np.array([161,155,84])
    high_red=np.array([179,255,255])

    red_mask=cv2.inRange(hsv_frame,low_red,high_red)
    red= cv2.bitwise_and(frame,frame,mask=red_mask)

#Lets frame on windows
    cv2.imshow("Frame", frame)
    cv2.imshow('Red',red)


# weight key event which is 1 and which is 27 then break the loop 

    key=cv2.waitKey(1)
    if key==27:
        break