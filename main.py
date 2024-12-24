import cv2
import os
import pickle
import numpy as np
import cvzone
from datetime import datetime
import json
import mediapipe as mp

# Load student data locally (using a JSON file)
try:
    with open('students.json', 'r') as file:
        students_data = json.load(file)
except FileNotFoundError:
    students_data = {}

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width of the frame
cap.set(4, 480)  # Height of the frame

# Graphics
imgBackground = cv2.imread('face_detection_attendance_system-main\\collage project\\Resources\\background.png')
folderModePath = 'face_detection_attendance_system-main\\collage project\\Resources\\Modes'

# Import images into a list
if folderModePath:
    modePathList = os.listdir(folderModePath)
    imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]
    print(f"Loaded {len(imgModeList)} mode images")
else:
    print("Error: Unable to read image from 'Resources\\Modes'")

# Load the encoded file
print("Loading the encoded file ...")
with open('face_detection_attendance_system-main\collage project\Encodefile.p', 'rb') as file:
    encodeknownwithid = pickle.load(file)
encodeknown, studentid = encodeknownwithid
print("File loaded!")

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Variables for modes and counters
modeType = 0
counter = 0
imgstudent = []

# Start Mediapipe Face Detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read the frame.")
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(img_rgb)
        imgBackground[162:162+480, 55:55+640] = img
        imgBackground[44:44+633, 808: 808+414] = imgModeList[modeType]

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                # Draw bounding box
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
           
                # Use bounding box as "encoding" for simplicity
                bbox_flat = np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).flatten()
                match_index = None
                min_distance = float('inf')

                # Compare with stored encodings
                for idx, stored_bbox in enumerate(encodeknown):
                    distance = np.linalg.norm(bbox_flat - np.array(stored_bbox))
                    if distance < min_distance:
                        min_distance = distance                 
                        match_index = idx

                if min_distance < 50:  # Adjust threshold as necessary
                    id = studentid[match_index]
                    if counter == 0:
                        counter = 1
                        modeType = 1
                else:
                    modeType = 3

                if counter != 0:
                    if counter == 1:
                        studentinfo = students_data.get(str(id), {})
                        print(studentinfo)

                        if studentinfo:
                            # Get the student's image (assuming the image is stored locally)
                            imgstudent = cv2.imread(f"face_detection_attendance_system-main\collage project\Images/{id}.png")

                            # Update attendance locally
                            studentinfo['total_attendance'] = studentinfo.get('total_attendance', 0) + 1
                            last_attendance = datetime.strptime(studentinfo.get('Last_attendance', "2024-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
                            seconds_elapsed = (datetime.now() - last_attendance).total_seconds()

                            if seconds_elapsed > 30:
                                studentinfo['Last_attendance'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            students_data[str(id)] = studentinfo  # Update data

                            # Save the updated student data locally (in JSON file)
                            with open('students.json', 'w') as file:
                                json.dump(students_data, file)

                        modeType = 3  # Go to idle mode

                    if modeType != 3:
                        if 10 < counter < 20:
                            modeType = 2

                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                        if counter <= 10:
                            cv2.putText(imgBackground, str(studentinfo.get('total_attendance', 0)), (861, 125),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                            cv2.putText(imgBackground, str(studentinfo.get('Branch', 'N/A')), (1006, 550),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(imgBackground, str(id), (1006, 493),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(imgBackground, str(studentinfo.get('semester', 'N/A')), (910, 625),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                            cv2.putText(imgBackground, str(studentinfo.get('current_year', 'N/A')), (1025, 625),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                            cv2.putText(imgBackground, str(studentinfo.get('starting_year', 'N/A')), (1125, 625),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

                            (w, h), _ = cv2.getTextSize(studentinfo.get('Name', 'N/A'), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                            offset = (414 - w) // 2
                            cv2.putText(imgBackground, str(studentinfo.get('Name', 'N/A')), (808 + offset, 445),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        imgstudent_resized = cv2.resize(imgstudent, (216, 216))
                        imgBackground[175: 175 + 216, 909: 909 + 216] = imgstudent_resized
                        counter += 1

                        if counter >= 20:
                            counter = 0
                            modeType = 0
                            studentinfo = {}
                            imgstudent = []
                            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
        else:
            modeType = 0
            counter = 0

        cv2.imshow("Face Attendance", imgBackground)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
