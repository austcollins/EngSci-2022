import cv2
import csv
import numpy as np
import timeit
import time
import datetime as dt
## todo: threading

video_filename = 'Participant34-VideoStress-Undistracted-bw.mov'
PresencerecordFILE1 = './sources/presence_log_'+video_filename+'.csv'


def appendCSV(file, array):
    with open(file, 'a', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(array)
    csvfile.close()

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
#cap = cv2.VideoCapture(0)
# To use a video file as input
cap = cv2.VideoCapture('./videos/'+video_filename)
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

is_assuming = False

is_present = False
present_times = []

# size of face allowed
size_of_face_allowed = [100, 100]

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
    # Detect the faces
    # defining distance from screen by limiting the size of the rectangle drawn
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, None, size_of_face_allowed)
    # Draw the rectangle around each face
    ## buffer
    if len(faces) == 0:
        frames_since_last_present = frames_processed - last_frame_present
        if absent_frames_buffer > frames_since_last_present:
            is_assuming = True
            face_assumed_present_count += 1
            for (x, y, w, h) in last_faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        elif is_assuming:
            if is_present == True:
                is_present = False
                print("now not-detected", frames_processed-absent_frames_buffer)
                videotime = (frames_processed-absent_frames_buffer)/video_fps
                present_times.append([videotime, is_present])
                appendCSV(PresencerecordFILE1, ["Timestamp", videotime, is_present])
            is_assuming = False
            face_assumed_present_count -= absent_frames_buffer
            print("deducted!")
            for (x, y, w, h) in last_faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 20)
    else:
        if is_present == False:
                is_present = True
                print("now detected", frames_processed)
                videotime = frames_processed/video_fps
                present_times.append([videotime, is_present])
                appendCSV(PresencerecordFILE1, ["timestamp", videotime, is_present])
        is_assuming = False
        last_faces = faces
        face_present_count += 1 #only count one frame, even if two faces
        last_frame_present = frames_processed
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            last_frame_present = frames_processed

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

percentage_time_present_but_distracted = time_assumed_present/total_video_time



print("===========")
percentage_time_present_string = "{:.0%}".format(percentage_time_present)
print("Percentage time present: %s"%percentage_time_present_string)
print("===========")
percentage_time_distracted_string = "{:.0%}".format(percentage_time_present_but_distracted)
print("Percentage assumed present (distracted maybe?): %s"%percentage_time_distracted_string)
print("===========")
print("Total video time processed: %s"%total_video_time)
print("Time assumed present: %s"%time_assumed_present)
print("Time face present: %s"%time_face_present)
print("Total time present: %s"%total_time_present)
print("===========")
print(present_times)

appendCSV(PresencerecordFILE1, [])
appendCSV(PresencerecordFILE1, ['percentage_time_present', percentage_time_present])
appendCSV(PresencerecordFILE1, ['total_video_time', total_video_time])
appendCSV(PresencerecordFILE1, ['time_assumed_present', time_assumed_present])
appendCSV(PresencerecordFILE1, ['time_face_present', time_face_present])
appendCSV(PresencerecordFILE1, ['total_time_present', total_time_present])
appendCSV(PresencerecordFILE1, ['percentage_time_assumed', percentage_time_present_but_distracted])


# Release the VideoCapture object
cap.release()