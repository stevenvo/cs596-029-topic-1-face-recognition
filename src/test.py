from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
import cv2


print(__doc__)

def plot_image(image, h, w):
    # """Helper function to plot a gallery of portraits"""
    # plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    # plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    # for i in range(n_row * n_col):
        # plt.subplot(n_row, n_col, i + 1)
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
        # plt.xticks(())
        # plt.yticks(())

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
# only consider people that have a minimum 70 pictures in the data set
# we only resize the images so that each have a 0.4 aspect ratio
lfw_people = fetch_lfw_people('../faces',min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape)
print(X_train)
