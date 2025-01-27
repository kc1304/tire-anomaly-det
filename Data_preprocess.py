import cv2
import numpy as np

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    # Dilation to enhance edges
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    return dilated

