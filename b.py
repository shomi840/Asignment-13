import cv2 as cv
import numpy as np
import random as rng

# Set seed for consistent random colors
rng.seed(12345)

# Callback for trackbar events
def thresh_callback(val):
    threshold = val

    # Apply Canny edge detection
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)

    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons + get bounding rectangles
    contours_poly = [cv.approxPolyDP(c, 3, True) for c in contours]
    boundRect = [cv.boundingRect(cp) for cp in contours_poly]

    # Create a black canvas for drawing
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # Draw only bounding rectangles
    for i in range(len(boundRect)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        x, y, w, h = boundRect[i]
        cv.rectangle(drawing, (x, y), (x + w, y + h), color, 2)

    # Show and save result
    cv.imshow('Bounding Boxes', drawing)
    save_path = r'C:\Users\ASUS\Desktop\New folder (10)\output_bounding_boxes.jpg'
    cv.imwrite(save_path, drawing)
    print(f"Contours detected: {len(contours)}")
    print(f"Result saved to: {save_path}")

# Load the input image
src = cv.imread(r'C:\Users\ASUS\Desktop\New folder (10)\red_block (1).jpg')
if src is None:
    print('Could not open or find the image.')
    exit(0)

# Convert to grayscale and apply blur
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))

# Display original image
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

# Create a trackbar for Canny threshold
max_thresh = 255
initial_thresh = 100
cv.createTrackbar('Canny thresh:', source_window, initial_thresh, max_thresh, thresh_callback)

# Run the initial detection
thresh_callback(initial_thresh)

# Wait until user presses any key
print("Press any key in the image window to exit...")
cv.waitKey(0)
cv.destroyAllWindows()
