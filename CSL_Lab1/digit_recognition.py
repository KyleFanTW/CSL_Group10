from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import cv2
# Use status bar to show the progress
import matplotlib.pyplot as plt

model = load_model('digit_classifier.h5')

#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Show data shape
#print('Training data shape:', train_images.shape)
#print('Training label shape:', train_labels.shape)

# A wrapper of KNN_classify where it reads a file and classify the test_image
def classify(k, test_image_path, test_image=None):
    if test_image is None:
        ori_img = cv2.imread(test_image_path, 0)
    else:
        ori_img = test_image
    cv2_img = cv2.resize(ori_img, (28, 28))
    label = model_classify(cv2_img)
    print('The label of the test image is:', label)

def model_classify(img):
    return np.argmax(model.predict(img))

def euclidean_dis(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

"""
def KNN_classify(k, test_image):
    label = 0 # The label we're giving to test_data
    test_image = test_image.flatten()
    centralize(test_image, mean_of_train)
    distance = [euclidean_dis(test_image, x_train) for x_train in train_images]
    sorted_index = np.argsort(distance)
    k_labels = train_labels[sorted_index[:k]]
    label_count = np.bincount(k_labels)
    label = np.argmax(label_count)
    return label
"""

def getXMean(data):
    # Get mean of all data
    data = np.reshape(data, (data.shape[0], -1))
    mean = np.mean(data, axis=0)
    return mean

def centralize(data, mean):
    # Centralize the data
    data = np.reshape(data, (data.shape[0], -1))
    data.astype(np.float64)
    data = data - mean
    return data

#mean_of_train = getXMean(train_images)
#train_images = centralize(train_images, mean_of_train)

# Load the MNIST dataset

# Create a simple window with a button classify
#import tkinter as tk

#root = tk.Tk()
#root.title('Digit Recognition')
#btn = tk.Button(root, text='Classify', command=lambda: classify(21, 'CSL_Group10\\CSL_Lab1\\path_image.png'))
#btn.pack()
#clear_btn = tk.Button(root, text='Clear and save', command= lambda: clear_save())
#root.mainloop()


