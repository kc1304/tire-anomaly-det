from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

from Data_preprocess import preprocess_image
import cv2

img = cv2.imread("C:/Users/aksha/Desktop/DIP_pr/Dataset/defective/Defective (16).jpg")

result = preprocess_image(img)
cv2.imshow("Result", result)
cv2.waitKey(0)