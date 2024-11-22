import cv2
import os
import time

import numpy as np


def train_recognizer(known_images_folder):
    """
    Train the LBPH face recognizer with known images.
    """
    # Initialize the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}

    # Loop through the known images folder
    for label, image_name in enumerate(os.listdir(known_images_folder)):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_path = os.path.join(known_images_folder, image_name)

            # Load the image and convert to grayscale
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces_in_image = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_in_image:
                # Add the face region and label
                faces.append(gray[y:y + h, x:x + w])
                labels.append(label)
                label_map[label] = image_name.split('.')[0]  # Using image name without extension as the label

    # Train the recognizer on the faces and labels
    recognizer.train(faces, np.array(labels))
    return recognizer, label_map


def mark_attendance(name):
    """
    Mark attendance by writing the name of the recognized person in the attendance log file.
    """
    with open("attendance_log.txt", "a") as file:
        file.write(f"{name} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Attendance marked for: {name}")


def face_recognition_system(known_images_folder):
    """
    Face recognition system to detect faces and log attendance.
    """
    # Train the recognizer with known faces
    recognizer, label_map = train_recognizer(known_images_folder)

    # Initialize the webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit the camera feed.")

    while True:
        # Capture each frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_in_frame:
            # Crop the face from the frame
            face_region = gray[y:y + h, x:x + w]

            # Predict the label (name) of the face
            label, confidence = recognizer.predict(face_region)

            name = "Unknown"
            if confidence < 100:
                name = label_map.get(label, "Unknown")

                # Mark attendance for recognized person
                mark_attendance(name)

            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame with face recognition
        cv2.imshow("Face Recognition Attendance System", frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    known_images_folder = "known_faces"  # Folder with known images (each named with the person's name)
    face_recognition_system(known_images_folder)
