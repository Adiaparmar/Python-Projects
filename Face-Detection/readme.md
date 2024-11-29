
# Face Recognition Attendance System

A simple face recognition-based attendance system built using Python and OpenCV. The system uses a webcam to capture images, train a face recognizer, and mark attendance automatically when a person is recognized in real-time.

## Features

- **Capture New Images**: Capture images of individuals and save them with their names.
- **Train Face Recognizer**: Train the face recognition model using the captured images.
- **Real-Time Face Recognition**: Detect faces in a webcam feed and mark attendance when a recognized person is found.
- **Attendance Logging**: Record the name and timestamp of recognized individuals in a log file.

## Requirements

- Python 3.x
- OpenCV (`opencv-python` and `opencv-contrib-python`)
- NumPy

You can install the required libraries using `pip`:

```bash
pip install opencv-python opencv-contrib-python numpy
```

## Setup

1. Clone this repository or download the project files to your local machine.

2. Install the required dependencies using `pip`:

```bash
pip install opencv-python opencv-contrib-python numpy
```

3. Ensure that your system has a working webcam connected for image capture and face detection.

4. Create a folder named `known_images_folder` to store the captured images. This folder will be used for training the face recognizer.

## How to Use

### 1. Capture a New Image
- Run the script.
- Select option 1 to **Capture a new image** using the webcam.
- When prompted, press **'c'** to capture an image and input the name for that person. The image will be saved with the entered name in the `known_images_folder`.

### 2. Train Face Recognizer
- Select option 2 to **Train the face recognizer**. The system will process the images in the `known_images_folder` to train the face recognition model.

### 3. Real-Time Face Recognition and Attendance Marking
- Select option 2 again to **Start face recognition and mark attendance**.
- The system will start capturing the video feed, detecting faces in real-time, and marking attendance for recognized faces by logging their names and timestamps in an `attendance_log.txt` file.

### 4. Exit the Program
- Select option 3 to **Exit** the program.

## Code Overview

- **capture_image()**: Captures images using the webcam and saves them with a user-defined name.
- **train_recognizer()**: Trains the LBPH face recognizer using images from the `known_images_folder`.
- **mark_attendance()**: Marks attendance by saving the recognized person's name and timestamp to a file.
- **face_recognition_system()**: The main system for real-time face recognition and attendance marking.
- **main_menu()**: The command-line menu that lets the user choose between capturing an image, starting the face recognition, or exiting the program.

## Output

- **attendance_log.txt**: A text file where the names and timestamps of recognized individuals are recorded.
- **Known Images Folder**: A folder where captured images are saved for training the recognizer.

## Potential Improvements

- **Multiple Face Recognition**: Handle multiple faces in the same frame.
- **Liveness Detection**: Add a feature to verify that the face is from a live person (not a photo).
- **Improved Recognition**: Use deep learning models for more accurate face recognition, especially in varying lighting conditions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- OpenCV for the computer vision capabilities.
- Python for building the application.
- LBPH (Local Binary Pattern Histogram) algorithm for face recognition.
