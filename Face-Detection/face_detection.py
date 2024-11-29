import cv2
import os
import time
import numpy as np


def capture_image(known_images_folder):
    """
    Capture an image using the webcam and save it to the known_faces folder.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'c' to capture an image, or 'q' to quit without capturing.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the camera feed
        cv2.imshow("Capture Image - Press 'c' to Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture the image
            # Ask for the name to label the image
            name = input("Enter the name for this image (no spaces): ").strip()
            if not name:
                print("Invalid name. Please try again.")
                continue

            # Save the image to the known_faces folder
            os.makedirs(known_images_folder, exist_ok=True)
            file_path = os.path.join(known_images_folder, f"{name}.jpg")
            cv2.imwrite(file_path, frame)
            print(f"Image saved as {file_path}.")
            break
        elif key == ord('q'):  # Quit without saving
            print("Image capture canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()


def train_recognizer(known_images_folder):
    """
    Train the LBPH face recognizer with known images.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}

    for label, image_name in enumerate(os.listdir(known_images_folder)):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            image_path = os.path.join(known_images_folder, image_name)

            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces_in_image = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_in_image:
                faces.append(gray[y:y + h, x:x + w])
                labels.append(label)
                label_map[label] = os.path.splitext(image_name)[0]

    if faces and labels:
        recognizer.train(faces, np.array(labels))
    else:
        print("No faces found in the known images folder. Please upload valid images.")

    return recognizer, label_map


def mark_attendance(name, marked_names):
    """
    Mark attendance by writing the name of the recognized person in the attendance log file.
    """
    if name not in marked_names:
        with open("attendance_log.txt", "a") as file:
            file.write(f"{name} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Attendance marked for: {name}")
        marked_names.add(name)


def face_recognition_system(known_images_folder):
    """
    Face recognition system to detect faces and log attendance.
    """
    recognizer, label_map = train_recognizer(known_images_folder)
    if not recognizer:
        print("Error: Unable to train recognizer. Returning to menu.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit the camera feed.")
    marked_names = set()  # Track attendance during the session

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces_in_frame = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_in_frame:
            face_region = gray[y:y + h, x:x + w]

            label, confidence = recognizer.predict(face_region)
            name = "Unknown"

            if confidence < 50:  # Adjust confidence threshold for better accuracy
                name = label_map.get(label, "Unknown")
                mark_attendance(name, marked_names)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_menu():
    known_images_folder = "known_images_folder"

    while True:
        print("\n--- Face Recognition System ---")
        print("1. Capture a new image using webcam")
        print("2. Start face recognition and mark attendance")
        print("3. Exit")

        try:
            choice = int(input("Enter your choice (1/2/3): "))
            if choice == 1:
                capture_image(known_images_folder)
            elif choice == 2:
                face_recognition_system(known_images_folder)
            elif choice == 3:
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number (1/2/3).")


if __name__ == "__main__":
    main_menu()
