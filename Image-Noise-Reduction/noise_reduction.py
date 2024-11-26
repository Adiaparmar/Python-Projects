import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds Gaussian noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Adds salt-and-pepper noise to an image."""
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(coords)] = 255

    # Add pepper
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(coords)] = 0

    return noisy_image


def apply_filters(image):
    """Applies various noise reduction filters."""
    avg_blur = cv2.blur(image, (5, 5))
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    median_blur = cv2.medianBlur(image, 5)
    bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

    return avg_blur, gaussian_blur, median_blur, bilateral_filter


def display_results(original, noisy, filters, filter_names):
    """Displays the original, noisy, and filtered images side by side."""
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    for i, (filtered_image, filter_name) in enumerate(zip(filters, filter_names)):
        plt.subplot(2, 3, i + 3)
        plt.title(filter_name)
        plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the image
    image_path = "images/sample.jpg"  # Replace with your image file path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
    else:
        print("Image loaded successfully!")

        # Add noise to the image
        noisy_image = add_gaussian_noise(image)  # You can replace this with add_salt_and_pepper_noise(image)

        # Apply filters to reduce noise
        avg_blur, gaussian_blur, median_blur, bilateral_filter = apply_filters(noisy_image)

        # Display results
        filter_names = ["Averaging Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter"]
        display_results(image, noisy_image, [avg_blur, gaussian_blur, median_blur, bilateral_filter], filter_names)
