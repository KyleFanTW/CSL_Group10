import cv2
import numpy as np
import os
import tkinter
import subprocess
# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture(1)

r_thres = 20
b_thres = 100
area_thres = 50

def show_image(path):
	img = cv2.imread(path)
	cv2.imshow('image', img)

def create_slider():
	cv2.namedWindow('slider')
	cv2.createTrackbar('R Threshold', 'slider', 0, 255, nothing)
	cv2.createTrackbar('B Threshold', 'slider', 0, 255, nothing)
	cv2.createTrackbar('Area Threshold', 'slider', 0, 100, nothing)


def nothing(x):
	pass

def get_slider_values():
	r_thres = cv2.getTrackbarPos('R Threshold', 'slider')
	b_thres = cv2.getTrackbarPos('B Threshold', 'slider')
	area_thres = cv2.getTrackbarPos('Area Threshold', 'slider')
	return r_thres, b_thres, area_thres

# Call a subprocess to run window
subprocess.Popen('python3 .\\ftir_image_processing.py', shell=True)

# Function that creates empty image with white background
def create_empty_image():
	width = int(cap.get(3))
	height = int(cap.get(4))
	return 255 * np.ones((height, width, 3), dtype="uint8")

clear_frame = 240
idle_frame = 0

path_image = None

# create_slider()

while(True):

	if path_image is None:
		# Create an empty image using width and height of camera
		width = int(cap.get(3))
		height = int(cap.get(4))
		path_image = create_empty_image()

	# Get one frame from the camera
	ret, frame = cap.read()

	# Split RGB channels

	b, g, r = cv2.split(frame)

	zeros = np.zeros(frame.shape[:2], dtype="uint8")

	# Perform thresholding to each channel

	# _, _, _ = get_slider_values()

	_, r = cv2.threshold(r, r_thres, 255, cv2.THRESH_BINARY)
	_, b_inv = cv2.threshold(b, b_thres, 255, cv2.THRESH_BINARY_INV)

	# Get the final result using bitwise operation
	
	ret = cv2.bitwise_and(r, b_inv)

	# Find and draw contours

	contours, hierarchy = cv2.findContours(ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	display = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
	cv2.drawContours(display, contours, -1, (0, 255, 0), 3)	

	# Iterate through each contour, check the area and find the center

	if len(contours) == 0:
		idle_frame += 1
		if idle_frame == clear_frame:
			print('idle: Clearing path_image and saving flipped version to current directory')
			#path_image = cv2.resize(path_image, (28, 28))
			# Change the color black to white, and red to black
			path_image = cv2.cvtColor(path_image, cv2.COLOR_BGR2GRAY)
			flipped_path_image = cv2.flip(path_image, 1)
			# Save the image to current directory
			cv2.imwrite('path_image.png', flipped_path_image)
			path_image = create_empty_image()
			idle_frame = 0
	else:
		idle_frame = 0

	for c in contours:
		area = cv2.contourArea(c)
		if area > area_thres:
			(x, y), radius = cv2.minEnclosingCircle(c)
			(x, y) = (int(x), int(y))
			radius = int(radius)
			cv2.circle(display, (x, y), radius, (0, 0, 255), 1)
			# Put text with the center and radius
			cv2.putText(display, "({}, {}) r={}".format(x, y, radius), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			# Draw a dot on (x,y) on the path_image
			cv2.circle(path_image, (x, y), 20, (0, 0, 0), -1)

	# Show the frame
	#cv2.imshow('frame', frame)
	#cv2.imshow('display', cv2.merge([zeros, zeros, ret]))
	flipped_path_image = cv2.flip(path_image, 1)
	cv2.imshow('path', flipped_path_image)

	# Press q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()