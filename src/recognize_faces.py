import os
import cv2
import sys
import re
import numpy as np

from glob import glob
from sklearn.decomposition import RandomizedPCA

DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING = '../data/training/'
h = 93
w = 93

def generate_eigenfaces():

    # Sample data for regex: '../data/training/01-Tracy-Cropped'
    # Extract [0]=01, [1] = Tracy
    def extract_id_name(path):
        print path
        p = re.compile(ur'^.*training\/(\d*)-(.*)-Cropped')
        search_obj = re.search(p, path)
        return search_obj.group(1), search_obj.group(2)


    def read_cropped_faces_from_files():
        X_train = []
        y_train = []
        for path in glob('../data/training/*-Cropped'):
            id, name = extract_id_name(path)
            files = glob("{0}/*.jpg".format(path))
            for f in files:
                X_train.append(cv2.imread(f))
                y_train.append(int(id))
        return X_train, y_train

    def compute_pca(X_train, y_train, n_components = 150):
        pca = RandomizedPCA(n_components, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((n_components, h, w))
        X_train_pca = pca.transform(X_train)


    X_train, y_train = read_cropped_faces_from_files()
    compute_pca(X_train, y_train)


generate_eigenfaces()
