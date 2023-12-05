import cv2
import numpy as np
from matplotlib import pyplot as plt

def wavelet_transform(image):
    # Perform wavelet transform to remove noise
    # You can use libraries like PyWavelets or OpenCV for this purpose
    # For simplicity, this example uses a simple smoothing filter
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

def differential_edge_detection(image):
    # Apply Differential edge detection
    edges = cv2.Laplacian(image, cv2.CV_64F)
    return edges

def log_edge_detection(image):
    # Apply LoG (Laplacian of Gaussian) edge detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    return edges

def canny_edge_detection(image):
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    return edges

def binary_morphology(image):
    # Apply Binary Morphology for edge detection
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    edges = dilation - erosion
    return edges

def bordering_closed(image):
    # Apply bordering closed to get clear and integral image profile
    # You may need to experiment with the structuring element and iterations
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed

def main():
    # Load the image
    image = cv2.imread("C:\\Users\\GAURAV TRIPATHI\\OneDrive\\Desktop\\ivp.jpg", cv2.IMREAD_GRAYSCALE)


    # Apply wavelet transform to remove noise
    denoised_image = wavelet_transform(image)

    # Apply different edge detection operators
    diff_edge = differential_edge_detection(denoised_image)
    log_edge = log_edge_detection(denoised_image)
    canny_edge = canny_edge_detection(denoised_image)
    binary_morphology_edge = binary_morphology(denoised_image)

    # Compare and visualize the results
    plt.figure(figsize=(12, 6))

    plt.subplot(231), plt.imshow(image, cmap="gray"), plt.title("Original Image")
    plt.subplot(232), plt.imshow(diff_edge, cmap="gray"), plt.title("Differential Edge Detection")
    plt.subplot(233), plt.imshow(log_edge, cmap="gray"), plt.title("LoG Edge Detection")
    plt.subplot(234), plt.imshow(canny_edge, cmap="gray"), plt.title("Canny Edge Detection")
    plt.subplot(235), plt.imshow(binary_morphology_edge, cmap="gray"), plt.title("Binary Morphology Edge Detection")

    # Apply bordering closed for clear and integral image profile
    closed_image = bordering_closed(binary_morphology_edge)
    plt.subplot(236), plt.imshow(closed_image, cmap="gray"), plt.title("Bordering Closed")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
