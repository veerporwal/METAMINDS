import mediapipe as mp
import cv2
import os
import pickle

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

folderPath = 'Images'  # Folder containing student images
studentid = []  # List to store student IDs (based on image filenames)
studentList = []  # List to store the student images

# Check if folder exists and load images
if os.path.exists(folderPath):
    modePathList = os.listdir(folderPath)
    for path in modePathList:
        if path.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            studentList.append(cv2.imread(os.path.join(folderPath, path)))
            studentid.append(os.path.splitext(path)[0])  # Student ID is the filename without extension
    print(f"Loaded {len(studentList)} images")
    print(f"Student IDs: {studentid}")
else:
    print("Error: Unable to read images from 'Images' folder")

# Function to extract face encodings from images
def find_encodings(imageList):
    encodeList = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        for img in imageList:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            results = face_detection.process(img_rgb)  # Detect faces
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box and keypoints as encoding
                    box = detection.location_data.relative_bounding_box
                    keypoints = detection.location_data.relative_keypoints
                    encoding = [box.xmin, box.ymin, box.width, box.height]
                    encoding.extend([kp.x for kp in keypoints])
                    encoding.extend([kp.y for kp in keypoints])
                    encodeList.append(encoding)  # Add encoding to the list
    return encodeList

# Start encoding
print("Encoding Started ...")
encoding_known = find_encodings(studentList)  # Generate encodings for all student images
encodeknownwithid = [encoding_known, studentid]  # Pair encodings with student IDs
print("Encoding Completed ...")

# Save encodings to a local file
with open('Encodefile.p', 'wb') as file:
    pickle.dump(encodeknownwithid, file)
print("File saved as 'Encodefile.p'!")
