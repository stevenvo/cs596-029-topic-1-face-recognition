import os
import cv2
import sys
import re
import numpy as np

from glob import glob
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING = '../data/training/'
DEFAULT_FACE_SIZE = 120.0

def generate_eigenfaces():

    # Sample data for regex: '../data/training/01-Tracy-Cropped'
    # Extract [0]=01, [1] = Tracy
    def extract_id_name(path):
        # print path
        p = re.compile(ur'^.*training\/(\d*)-(.*)-Cropped')
        search_obj = re.search(p, path)
        return search_obj.group(1), search_obj.group(2)


    # return is an array of faces in reshape (1,-1) format
    def read_cropped_faces_from_files():
        X_train = []
        y_train = []
        for path in glob('../data/training/*-Cropped'):
            id, name = extract_id_name(path)
            files = glob("{0}/*.jpg".format(path))
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = img.flatten()
                X_train.append(img)
                y_train.append(int(id))
        return X_train, y_train


    def compute_pca(X_train, y_train, n_components = 30):
        # X_train = [[1, 2, 3], [4, 5, 6]]
        # print len(X_train)
        pca = RandomizedPCA(n_components, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((n_components, DEFAULT_FACE_SIZE, DEFAULT_FACE_SIZE))
        # print eigenfaces.shape
        # for eigenface in eigenfaces:
            # eigenface = eigenface * 1000000 + 100
            # cv2.imshow("eigenface", eigenface)
            # cv2.waitKey(0)
        X_train_pca = pca.transform(X_train)
        return pca, X_train_pca


    def train_model(pca, X_train_pca, y_train):
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        print("Best estimator found by grid search: {0}".format(clf.best_estimator_))
        return clf

    X_train, y_train = read_cropped_faces_from_files()
    pca, X_train_pca = compute_pca(X_train, y_train)
    clf = train_model(pca, X_train_pca, y_train)


generate_eigenfaces()
