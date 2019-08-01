# -*- coding: utf-8 -*-

#Building a happiness detector

# importing the libraries
import cv2

#loading the cascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# creating a function that will do the detections
def detect(gray_frame, original_frame):
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(original_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_org = original_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.5, 22)
        for(sx, sy, sw, sh) in smile:    
            cv2.rectangle(roi_org, (sx, sy), (sx+sw, sy+sh), (0,0,0), 2)
            print("USER HAS SMILED or is SMILING")
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_org, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
    return original_frame

#Facial recognition detection

video_capature = cv2.VideoCapture(0)

while True:
    _, frame = video_capature.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capature.release()
cv2.destroyAllWindows()
