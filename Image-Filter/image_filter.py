# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
#
# def display_image(window_title, image, is_gray=False):
#     """
#     Utility function to display an image using Matplotlib.
#     """
#     if is_gray:
#         plt.imshow(image, cmap="gray")
#     else:
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title(window_title)
#     plt.axis("off")
#     plt.show()
#
#
# def apply_grayscale(image):
#     """
#     Convert the image to grayscale.
#     """
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     display_image("Grayscale Image", gray_image, is_gray=True)
#
#
# def apply_gaussian_blur(image):
#     """
#     Apply Gaussian blur to the image.
#     """
#     blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
#     display_image("Gaussian Blurred Image", blurred_image)
#
#
# def apply_edge_detection(image):
#     """
#     Apply edge detection (Canny) to the image.
#     """
#     edges = cv2.Canny(image, 100, 200)
#     display_image("Edge Detection", edges, is_gray=True)
#
#
# def resize_image(image):
#     """
#     Resize the image to 300x300 pixels.
#     """
#     resized_image = cv2.resize(image, (300, 300))
#     display_image("Resized Image", resized_image)
#
#
# def rotate_image(image):
#     """
#     Rotate the image by 45 degrees.
#     """
#     height, width = image.shape[:2]
#     center = (width // 2, height // 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
#     display_image("Rotated Image", rotated_image)
#
#
# def flip_image(image):
#     """
#     Flip the image horizontally.
#     """
#     flipped_image = cv2.flip(image, 1)
#     display_image("Flipped Image", flipped_image)
#
#
# def apply_filters(image):
#     """
#     Interactive menu for applying filters and transformations to the image.
#     """
#     while True:
#         print("\nChoose an option:")
#         print("1. Grayscale")
#         print("2. Gaussian Blur")
#         print("3. Edge Detection")
#         print("4. Resize")
#         print("5. Rotate")
#         print("6. Flip")
#         print("0. Exit")
#
#         try:
#             choice = int(input("Enter your choice: "))
#         except ValueError:
#             print("Invalid input! Please enter a number.")
#             continue
#
#         if choice == 1:
#             apply_grayscale(image)
#         elif choice == 2:
#             apply_gaussian_blur(image)
#         elif choice == 3:
#             apply_edge_detection(image)
#         elif choice == 4:
#             resize_image(image)
#         elif choice == 5:
#             rotate_image(image)
#         elif choice == 6:
#             flip_image(image)
#         elif choice == 0:
#             print("Exiting the program.")
#             break
#         else:
#             print("Invalid choice! Please select a valid option.")
#
#
# if __name__ == "__main__":
#     # Load the input image
#     image_path = "images/sample.jpg"  # Replace with your image file path
#     image = cv2.imread(image_path)
#
#     if image is None:
#         print("Error: Unable to load the image. Please check the file path.")
#     else:
#         print("Image loaded successfully!")
#         display_image("Original Image", image)
#         apply_filters(image)


import cv2
import matplotlib.pyplot as plt


def display_image(window_title, image, is_gray=False):
    """
    Utility function to display an image using Matplotlib.
    """
    if is_gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(window_title)
    plt.axis("off")
    plt.show()


def apply_grayscale(image):
    """
    Convert the image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(image):
    """
    Apply Gaussian blur to the image.
    """
    return cv2.GaussianBlur(image, (15, 15), 0)


def apply_edge_detection(image):
    """
    Apply edge detection (Canny) to the image.
    """
    return cv2.Canny(image, 100, 200)


def resize_image(image):
    """
    Resize the image to 300x300 pixels.
    """
    return cv2.resize(image, (300, 300))


def rotate_image(image):
    """
    Rotate the image by 45 degrees.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def flip_image(image):
    """
    Flip the image horizontally.
    """
    return cv2.flip(image, 1)


def apply_filters_on_frame(frame):
    """
    Apply filters to the live frame.
    """
    # Display the interactive menu to apply a specific filter
    print("\nChoose an option:")
    print("1. Grayscale")
    print("2. Gaussian Blur")
    print("3. Edge Detection")
    print("4. Resize")
    print("5. Rotate")
    print("6. Flip")
    print("0. Exit")

    try:
        choice = int(input("Enter your choice: "))
    except ValueError:
        print("Invalid input! Please enter a number.")
        return frame

    if choice == 1:
        frame = apply_grayscale(frame)
        display_image("Grayscale Image", frame, is_gray=True)
    elif choice == 2:
        frame = apply_gaussian_blur(frame)
        display_image("Gaussian Blurred Image", frame)
    elif choice == 3:
        frame = apply_edge_detection(frame)
        display_image("Edge Detection", frame, is_gray=True)
    elif choice == 4:
        frame = resize_image(frame)
        display_image("Resized Image", frame)
    elif choice == 5:
        frame = rotate_image(frame)
        display_image("Rotated Image", frame)
    elif choice == 6:
        frame = flip_image(frame)
        display_image("Flipped Image", frame)
    elif choice == 0:
        print("Exiting the program.")
        return None
    else:
        print("Invalid choice! Please select a valid option.")

    return frame


def start_camera():
    """
    Open the camera and apply filters in real-time.
    """
    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit the camera feed.")
    while True:
        ret, frame = cap.read()  # Capture a frame from the camera

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Apply filters to the captured frame
        frame = apply_filters_on_frame(frame)
        if frame is None:
            break  # Exit if user selects '0' (Exit)

        # Show the original live frame
        cv2.imshow("Live Camera Feed", frame)

        # Break the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
