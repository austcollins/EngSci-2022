import cv2
import numpy as np
import timeit


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
#cap = cv2.VideoCapture(0)
# To use a video file as input
cap = cv2.VideoCapture('test.mov')

video_fps = cap.get(cv2.CAP_PROP_FPS)

face_present_count = 0

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    if img is None:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    # defining distance from screen by limiting the size of the rectangle drawn
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, None, [250, 250])
    # Draw the rectangle around each face

    for (x, y, w, h) in faces:
        face_present_count += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    #cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break


total_time_present = face_present_count/video_fps

print(total_time_present)

# Release the VideoCapture object
cap.release()