# Image Noise Reduction and Evaluation

## Objective
The goal of this project is to study the effects of various types of noise on digital images and implement techniques to reduce noise while maintaining the quality of the original image. The project also evaluates the performance of these noise reduction techniques using quantitative metrics like **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**.

---

## Key Features

### 1. Noise Addition
The program simulates real-world image degradation by adding different types of noise to an image:
- **Gaussian Noise**: Models random variations in brightness.
- **Salt-and-Pepper Noise**: Randomly introduces white (salt) and black (pepper) pixels.
- **Speckle Noise**: Simulates noise proportional to pixel intensity.
- **Motion Blur**: Mimics the effect of movement during image capture.

### 2. Noise Reduction Filters
The program applies the following noise reduction techniques:
- **Averaging Filter**: Reduces noise by averaging pixel values in a local neighborhood.
- **Gaussian Filter**: Uses a Gaussian function for blurring, effective for Gaussian noise.
- **Median Filter**: Removes noise by replacing each pixel with the median value in its neighborhood, ideal for salt-and-pepper noise.
- **Bilateral Filter**: Smoothens images while preserving edges by combining spatial and intensity information.

### 3. Performance Evaluation
The effectiveness of noise reduction is assessed using:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the difference between the original and filtered images.
- **SSIM (Structural Similarity Index)**: Evaluates perceptual quality by comparing structures in the images.

### 4. Visualization
The results are displayed in a grid format, showing:
- The original image.
- The noisy image.
- Filtered images alongside their PSNR and SSIM values.

### 5. Image Saving
All noisy and filtered images are saved to a specified output folder for later analysis.

---

## Implementation Details

### Programming Language
Python

### Libraries Used
- **OpenCV**: For image processing operations.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualizing the results.
- **Scikit-Image**: For calculating PSNR and SSIM.

### Workflow
1. Load an input image.
2. Apply a noise type to simulate image degradation.
3. Use various filtering techniques to reduce the noise.
4. Calculate PSNR and SSIM to evaluate filter performance.
5. Display and save the results.

---

## Challenges and Solutions

### SSIM Error for Small Images
- **Challenge**: The SSIM calculation window size was larger than some image dimensions.
- **Solution**: Dynamically adjusted the `win_size` parameter for smaller images.

### Filter Kernel Size
- **Challenge**: Optimal kernel size for each filter varies based on image resolution.
- **Solution**: Made kernel size customizable to fine-tune results.

---

## Applications
1. **Photography and Videography**:
   - Reducing noise in photos or videos captured in low-light conditions.
2. **Medical Imaging**:
   - Enhancing the clarity of noisy CT or MRI scans.
3. **Satellite and Remote Sensing**:
   - Improving the quality of images degraded by atmospheric interference.

---

## Results and Observations
- The project demonstrates that:
  - **Median Filters** are most effective for salt-and-pepper noise.
  - **Gaussian Filters** work best for Gaussian noise.
  - **Bilateral Filters** preserve edges better than other methods.
- **PSNR and SSIM** provide quantitative confirmation of these observations.

---

## Conclusion
This project offers a comprehensive framework for understanding and mitigating noise in digital images. By integrating advanced noise models and noise reduction techniques, it provides both a practical tool and a platform for further research in the field of image processing.

---
