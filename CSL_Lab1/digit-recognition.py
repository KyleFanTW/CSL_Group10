from keras.datasets import mnist
import numpy as np

# Use status bar to show the progress
from tqdm import tqdm
import matplotlib.pyplot as plt

# A wrapper of KNN_classify where it reads a file and classify the test_image
def classify(k, train_data, train_labels, test_image):
    # Read the image
    image = plt.imread(test_image)
    # Reshape image to 28x28
    image = np.reshape(image, (28, 28))
    # Convert the image to grayscale
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    # Show the image
    plt.imshow(image, cmap='gray')
    plt.show()
    # Classify the image
    label = KNN_classify(k, train_data, train_labels, image)
    print('The label of the test image is:', label)

def KNN_classify(k, train_data, train_labels, test_image):
    # Use euclidean distance to find the nearest neighbor
    num_train = train_data.shape[0] # The number of training data
    label = 0 # The label we're giving to test_data

    # Reshape the image
    test_image = test_image.reshape(1, -1)
    print('Start KNN classification')

    # Calculate the distance between test_image and all training data
    distance = np.sqrt(np.sum(np.square(train_data - test_image), axis=1))
    # Sort the distance and get the index of the first k nearest neighbor
    sorted_index = np.argsort(distance)
    # Get the label of the first k nearest neighbor
    k_labels = train_labels[sorted_index[:k]]
    # Count the number of each label
    label_count = np.bincount(k_labels)
    # Get the label with the most count
    label = np.argmax(label_count)

    print('KNN classification finished')

    return label

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

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Show data shape
print('Training data shape:', train_images.shape)
print('Training label shape:', train_labels.shape)


mean_of_train = getXMean(train_images)
train_images = centralize(train_images, mean_of_train)

# Create a simple window with a button classify
import tkinter as tk

root = tk.Tk()
root.title('Digit Recognition')
btn = tk.Button(root, text='Classify', command=lambda: classify(3, train_images, train_labels, 'test_image.png'))
btn.pack()
root.mainloop()


