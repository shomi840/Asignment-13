import cv2 as cv
import numpy as np
import random as rng

# Set seed for consistent colors
rng.seed(12345)

# Callback for trackbar
def thresh_callback(val):
    threshold = val
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [cv.approxPolyDP(c, 3, True) for c in contours]
    boundRect = [cv.boundingRect(cp) for cp in contours_poly]
    centers = [cv.minEnclosingCircle(cp)[0] for cp in contours_poly]
    radius = [cv.minEnclosingCircle(cp)[1] for cp in contours_poly]

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours_poly, i, color)
        x, y, w, h = boundRect[i]
        cv.rectangle(drawing, (x, y), (x + w, y + h), color, 2)
        cx, cy = int(centers[i][0]), int(centers[i][1])
        cv.circle(drawing, (cx, cy), int(radius[i]), color, 2)

    cv.imshow('Contours', drawing)
    # Optional: save the result image
    cv.imwrite(r'C:\Users\ASUS\Desktop\New folder (10)\flower_contours.jpg', drawing)

# Load the image (✔ make sure the file exists)
src = cv.imread(r'C:\Users\ASUS\Desktop\New folder (10)\flower.jpg')
if src is None:
    print('Could not open or find the image.')
    exit(0)

# Convert to grayscale and blur
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

# Display the original image
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

# Create a trackbar to control Canny threshold
max_thresh = 255
initial_thresh = 100
cv.createTrackbar('Canny thresh:', source_window, initial_thresh, max_thresh, thresh_callback)

# Run initial detection
thresh_callback(initial_thresh)

# Wait for a key press and close all windows
print("Press any key in the image window to exit...")
cv.waitKey(0)
cv.destroyAllWindows()
