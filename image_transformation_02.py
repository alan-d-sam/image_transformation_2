import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    inverse_image = cv2.bitwise_not(image)
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(grayscale)
    contrast_stretched = cv2.normalize(image, None, -100, 255, cv2.NORM_MINMAX)
    
    hist_eq_image = cv2.equalizeHist(grayscale)
    
    edges = cv2.Canny(grayscale, 100, 200)
    
    cv2.imshow("Original Image", image)
    cv2.imshow("Inverse Image", inverse_image)
    cv2.imshow("Contrast Stretched Image", contrast_stretched)
    cv2.imshow("Histogram Equalized Image", hist_eq_image)
    cv2.imshow("Edge Detection", edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "car.jpg"
process_image(image_path)
