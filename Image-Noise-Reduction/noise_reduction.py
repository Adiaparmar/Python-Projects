
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

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

def add_speckle_noise(image, mean=0, sigma=0.1):
    """Adds speckle noise to an image."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + image.astype(np.float32) * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_motion_blur(image, kernel_size=15):
    """Adds motion blur to an image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    motion_blur = cv2.filter2D(image, -1, kernel)
    return motion_blur

def apply_filters(image, kernel_size=5):
    """Applies various noise reduction filters."""
    avg_blur = cv2.blur(image, (kernel_size, kernel_size))
    gaussian_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    median_blur = cv2.medianBlur(image, kernel_size)
    bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)

    return avg_blur, gaussian_blur, median_blur, bilateral_filter

def evaluate_performance(original, filtered):
    """Calculates PSNR and SSIM between the original and filtered images."""
    psnr_value = psnr(original, filtered, data_range=255)
    try:
        ssim_value = ssim(original, filtered, multichannel=True, data_range=255, win_size=3)
    except ValueError:
        ssim_value = ssim(original, filtered, multichannel=True, data_range=255, win_size=1)
    return psnr_value, ssim_value


def display_results(original, noisy, filters, filter_names, original_image):
    """Displays the original, noisy, and filtered images with performance metrics."""
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
        psnr_value, ssim_value = evaluate_performance(original_image, filtered_image)
        plt.subplot(2, 3, i + 3)
        plt.title(f"{filter_name}\nPSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")
        plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def save_results(noisy, filters, filter_names, output_folder="output"):
    """Saves the noisy and filtered images."""
    os.makedirs(output_folder, exist_ok=True)

    cv2.imwrite(f"{output_folder}/noisy_image.jpg", noisy)
    for filtered_image, filter_name in zip(filters, filter_names):
        filename = f"{output_folder}/{filter_name.replace(' ', '_').lower()}.jpg"
        cv2.imwrite(filename, filtered_image)
    print(f"Results saved in '{output_folder}' folder.")

if __name__ == "__main__":
    # Load the image
    image_path = "images/sample.jpg"  # Replace with your image file path
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image. Please check the file path.")
    else:
        print("Image loaded successfully!")

        # Add noise to the image
        noisy_image = add_gaussian_noise(image)  # Replace with other noise functions if needed

        # Apply filters to reduce noise
        avg_blur, gaussian_blur, median_blur, bilateral_filter = apply_filters(noisy_image, kernel_size=7)

        # Display results
        filter_names = ["Averaging Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter"]
        display_results(image, noisy_image, [avg_blur, gaussian_blur, median_blur, bilateral_filter], filter_names, image)

        # Save results
        save_results(noisy_image, [avg_blur, gaussian_blur, median_blur, bilateral_filter], filter_names)
