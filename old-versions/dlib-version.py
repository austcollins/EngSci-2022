import cv2
import dlib
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
cap = cv2.VideoCapture('./videos/Austin-Distracted-Video-2.mov')
video_fps = cap.get(cv2.CAP_PROP_FPS)

# total frames in the video
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# how much of the video to process
process_all_frames = True
seconds_to_process = 20 ## change above to false to process part of the video

frames_to_process = seconds_to_process*video_fps
if process_all_frames:
    frames_to_process = total_frames
# number of frames which faces have been detected
face_present_count = 0
face_assumed_present_count = 0

# the last percentage shown (progress tracker)
last_percent = ""

# the last frame in which a face was detected
last_frame_present = 0

# number of frames the face has to be missing before they are counted as absent
buffer_seconds = 2
absent_frames_buffer = video_fps*buffer_seconds

# last faces (used for buffer)
last_faces = []

# size of face allowed
size_of_face_allowed = [150, 150]

# calculating progress
start_time = time.time()
frames_processed = 0
def calcProcessTime(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime)
    
#eye detection for dlib
predictor = dlib.shape_predictor('./training-files/shape_predictor_68_face_landmarks.dat')
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# main loop
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    if img is None:
        break

    #only analyse some of the video
    if frames_processed>(frames_to_process):
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #dlib version
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1) # rects contains all the faces detected

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    #lets see how we're doing
    frames_processed += 1
    progress = frames_processed/frames_to_process
    #if (progress % 0.01) == 0:
    percentage = "{:.0%}".format(progress)
    if percentage != last_percent:
        print(percentage)
        last_percent = percentage
        timeleft = calcProcessTime(start_time, frames_processed, frames_to_process)
        print("time elapsed: %s(s), time left: %s(s), estimated finish time: %s"%timeleft)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

time_assumed_present = face_assumed_present_count/video_fps
time_face_present = face_present_count/video_fps
total_time_present = time_assumed_present+time_face_present

total_video_time = frames_processed/video_fps
percentage_time_present = total_time_present/total_video_time

print("===========")
percentage_time_present_string = "{:.0%}".format(percentage_time_present)
print("Percentage time present: %s"%percentage_time_present_string)
print("===========")
print("Total video time processed: %s"%total_video_time)
print("Time assumed present: %s"%time_assumed_present)
print("Time face present: %s"%time_face_present)
print("Total time present: %s"%total_time_present)

# Release the VideoCapture object
cap.release()