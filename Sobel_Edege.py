import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import itertools
#from cv2 import addWeighted

# Load the image in grayscale
Morp = cv2.imread("Billeder/IMG_20251104_094347.jpg", cv2.IMREAD_GRAYSCALE)

# Resize the image 
Morp = cv2.resize(Morp, (800, 600))  # Adjust width and height as needed

# Display the original image 
cv2.imshow("Original Image", Morp)

# Apply Gaussian blur
Gauss = cv2.GaussianBlur(Morp, (5, 5), 1)

# Define Sobel kernels
kernal_Gx = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)
kernal_Gy = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]], dtype=np.float32)

# Apply Sobel filters
Blurx = cv2.filter2D(Gauss, -1, kernal_Gx)
Blury = cv2.filter2D(Gauss, -1, kernal_Gy)

# Combine the results
Full = cv2.addWeighted(Blurx, 0.5, Blury, 0.5, 0)

# Display the edge-detected image
cv2.imshow("Sobel Edges", Full)
cv2.waitKey(0)
cv2.destroyAllWindows()
