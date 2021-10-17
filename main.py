import sys
import cv2
from random import *

#Load pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def Webcam():
    #Choose an image to detect faces in
    webcam = cv2.VideoCapture(0)

    while True:
        successful_frame_read, frame = webcam.read()

        frame = cv2.flip(frame, 1)

        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = trained_data.detectMultiScale(grayscaled_img)

        # Draw rectangle around faces
        i=0
        for face in face_coordinates:
            (x, y, w, h) = face_coordinates[i]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, 'Detected face ' + str(i+1), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            i += 1

        cv2.imshow('opencv Face Detection', frame)
        key = cv2.waitKey(1)

        if key==81 or key==113:
            break
    webcam.release()

def Image(img):
    img = cv2.imread(img)

    #Convert image to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect where the face is
    face_coordinates = trained_data.detectMultiScale(grayscaled_img)

    # Draw rectangle around faces
    i=0
    for face in face_coordinates:
        (x, y, w, h) = face_coordinates[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(50,256), randrange(50,256), randrange(50,256)), 3)
        cv2.putText(img, 'Detected face ' + str(i+1), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        i += 1

    #Show the opencv image
    cv2.imshow('opencv Face Detector', img)

    #Wait until a key is pressed
    cv2.waitKey()

if len(sys.argv) > 1:
    if sys.argv[1] == "--image":
        image = sys.argv[2]
        Image(image)

    elif sys.argv[1] == "--webcam":
        Webcam()
    else:
        print("Must use --image [path to image] or --webcam")
else:
    print("Must use --image [path to image] or --webcam")

print("---END---")