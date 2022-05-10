import cv2
import numpy as np
import timeit
import time
import datetime as dt
## todo: threading

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
#cap = cv2.VideoCapture(0)
# To use a video file as input
cap = cv2.VideoCapture(0)

# number of frames which faces have been detected
face_present_count = 0
face_assumed_present_count = 0
frames_processed = 0

# the last frame in which a face was detected
last_frame_present = 0

# number of frames the face has to be missing before they are counted as absent
absent_frames_buffer = 30

# last faces (used for buffer)
last_faces = []

# size of face allowed
size_of_face_allowed = [150, 150]

# main loop
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    # defining distance from screen by limiting the size of the rectangle drawn
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, None, size_of_face_allowed)
    # Draw the rectangle around each face

    ## buffer
    if len(faces) == 0:
        frames_since_last_present = frames_processed - last_frame_present
        if absent_frames_buffer > frames_since_last_present:
            face_assumed_present_count += 1
            for (x, y, w, h) in last_faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if absent_frames_buffer === frames_since_last_present:
            face_assumed_present_count -= absent_frames_buffer
            for (x, y, w, h) in last_faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    else:
        last_faces = faces
        face_present_count += 1 #only count one frame, even if two faces
        last_frame_present = frames_processed
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)
            last_frame_present = frames_processed

    #lets see how we're doing
    frames_processed += 1

    percentage_time_present = (face_assumed_present_count+face_present_count)/frames_processed
    percentage_time_present_string = "{:.0%}".format(percentage_time_present)
    print("Percentage time present: %s"%percentage_time_present_string)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()