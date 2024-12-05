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

# Plot 1: Original Image
ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
ax[0].set_title("Original Image")
ax[0].axis("off")

# Plot 2: Hue Histogram
ax[1].plot(signal, color='orange', lw=2)
ax[1].set_title("Histogram of Values")
ax[1].set_xlabel("Value")
ax[1].set_ylabel("Frequency gradient")
ax[1].axvline(treshold, color='green', linestyle='--', label=f'Treshold = {treshold:.2f}')

ax[1].grid(alpha=0.5)
ax[1].legend()

# Plot 3: Dilated Binary Mask
ax[2].imshow(filtered_mask, cmap='gray')
ax[2].set_title("Binary Mask after processing")
ax[2].axis("off")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
