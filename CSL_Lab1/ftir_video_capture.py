import cv2
import numpy as np
import os
import tkinter
import subprocess
import math

from digit_recognition import classify
# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture(0)

r_thres = 20
b_thres = 0
area_thres = 15

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
# subprocess.Popen('python3 ftir_image_processing.py', shell=True)

# Function that creates empty image with white background
def create_empty_image(dim):
	return np.zeros((dim, dim, 3), dtype="uint8")

path_image = None

def digit_zoom(img, padding):
	contours, hierarchy = cv2.findContours(cv2.cvtColor(flipped_path_image, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Find contour of the maximum area
	if len(contours) == 0:
		return img
	sort_index = np.argsort([cv2.contourArea(c) for c in contours])
	c = contours[sort_index[-1]]
	(x, y), radius = cv2.minEnclosingCircle(c)
	(x, y) = (int(x), int(y))
	radius = int(radius+padding)

	# Take only the image inside the circle
	zoomed_image = img[y-radius:y+radius, x-radius:x+radius]

	return zoomed_image
	

def flip_save():
	flipped_path_image = cv2.flip(path_image, 1)
	cv2.imwrite('path_image.png', digit_zoom(flipped_path_image, 50))
	

flag = 0

def clear_save():
	print('[clear_save()]: Clearing path_image and saving flipped version to current directory')
	flip_save()
	flag = 0
	return create_empty_image(cropped_image_dim)

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

clear_frame = 240
sample_frame = 10
frame_counter = 0

cropped_image_dim = min(int(cap.get(3)), int(cap.get(4)))

# create_slider()

px = 0
py = 0

while(True):

	frame_counter += 1 # For sampling

	if path_image is None:
		# Create an empty image using width and height of camera
		path_image = create_empty_image(cropped_image_dim)
	# Get one frame from the camera
	# Crop the frames
	
	ret, ori_frame = cap.read()
	frame = ori_frame[:cropped_image_dim][:cropped_image_dim]
	

	# Split RGB channels

	b, g, r = cv2.split(frame)

	zeros = np.zeros(frame.shape[:2], dtype="uint8")

	# Perform thresholding to each channel

	# r_thres, b_thres, area_thres = get_slider_values()

	_, r = cv2.threshold(r, r_thres, 255, cv2.THRESH_BINARY)
	_, b_inv = cv2.threshold(b, b_thres, 255, cv2.THRESH_BINARY_INV)

	# Get the final result using bitwise operation
	
	ret = cv2.bitwise_and(r, b_inv)

	# Find and draw contours

	contours, hierarchy = cv2.findContours(ret, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	display = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)
	cv2.drawContours(display, contours, -1, (0, 255, 0), 3)	

	# Iterate through each contour, check the area and find the center


	if frame_counter > sample_frame:
		frame_counter = 0
		flip_save()
		classify(21, 'path_image.png')

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
			# cv2.circle(path_image, (x, y), 10, (255,255,255), -1)
			if flag == 1 and euclidean_distance((px,py),(x,y)) < 100:
				cv2.line(path_image, (px,py), (x,y), (0,0,255), 18)
			elif flag == 0:
				flag = 1
			px = x
			py = y


	
	# Show the frame
	cv2.imshow('frame', frame)
	#cv2.imshow('display', display)
	#cv2.imshow('red', cv2.merge([zeros, zeros, ret]))
	flipped_path_image = cv2.flip(path_image, 1)
	cv2.imshow('path', flipped_path_image)

	
	# Press q to quit
	key_ret = cv2.waitKey(100) & 0xFF
	if key_ret == ord('q'):
		break
	elif key_ret == ord('c'):
		flip_save()
		classify(21, 'path_image.png')
	elif key_ret == ord('s'):
		path_image = clear_save()

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
