import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt

image_path = os.path.join("Photos/Tymio_islands_resised.jpg")
# image_path = os.path.join("pictures", "warped.jpg")

original = cv2.imread(image_path)
# image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
image = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)

hist = cv2.calcHist([image[:,:,0]], [0], None, [180], [0, 180])  # 180 bins for Hue in the range [0, 180]

signal = np.gradient(medfilt(hist.flatten(), 15))
higest = 125

grad_threshold = -20
highest = 125

# Iterate through the gradient to find the highest index crossing the threshold
crossing_index = -1
for i in range(highest - 1, 1, -1):  # Start from the index just below 'highest' and go backward
    if signal[i] < grad_threshold:
        crossing_index = i
        break

if crossing_index != -1:
    print(f"The highest index where the signal crosses {grad_threshold} is {crossing_index}.")
else:
    print(f"No crossing found before index {highest}.")
    
treshold = crossing_index
mask = image[:,:,0] < treshold
obstacles = np.uint8(mask * 255)


# Minimum size threshold for connected components
COUNT = 3000
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obstacles, connectivity=8)
filtered_mask = np.zeros_like(obstacles)
for i in range(1, num_labels):  # Skip label 0 (background)
    if stats[i, cv2.CC_STAT_AREA] >= COUNT:
        filtered_mask[labels == i] = 255
        
# smoothing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 3x3 square kernel
filtered_mask = cv2.dilate(filtered_mask, kernel)
filtered_mask = cv2.erode(filtered_mask, kernel)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Invert the colors and display the image in full window mode
inverted_mask = cv2.bitwise_not(filtered_mask)  # Invert the colors (black to white and vice versa)

# Configure the display for full-window mode
plt.figure(figsize=(12, 12))  # Adjust size for a larger display
plt.imshow(inverted_mask, cmap='gray')  # Use 'gray' colormap for inverted binary mask
plt.axis("off")  # Remove axes for a cleaner look
plt.tight_layout()  # Use tight layout for full coverage of display area
plt.show()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()