from pudb import set_trace

import os
import cv2
import sys
import re
import numpy as np
import pandas as pd

from glob import glob
from time import time
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score


CROPPED_IMAGES_DIRECTORY = '../data/{0}/*-Cropped'
DEFAULT_FACE_SIZE = 120.0
TARGET_NAMES = ['Tracy', 'Trish', 'Steven']
TRAINING = 'training'
TESTING = 'testing'
N_COMPONENTS = 30

def recognize_face():

    # Sample data for regex: '../data/training/01-Tracy-Cropped'
    # Extract [0]=01, [1] = Tracy
    def extract_id_name(path):
        # print path
        p = re.compile(ur'^.*\/(\d*)-(.*)-Cropped')
        search_obj = re.search(p, path)
        return search_obj.group(1), search_obj.group(2)

    # return is an array of faces in reshape (1,-1) format
    def read_cropped_faces(mode):
        X = []
        y = []
        dir_path = CROPPED_IMAGES_DIRECTORY.format(mode)
        print "\n======LOAD {0} DATA=======".format(mode.upper())
        for path in glob(dir_path):
            id, name = extract_id_name(path)
            files = glob("{0}/*.jpg".format(path))
            print "- Mode, id, name, total images: {0}, {1}, {2}, {3}".format(mode, id, name, len(files))
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = img.flatten()
                X.append(img)
                y.append(int(id))
        return X, y

    def compute_pca(X_train, y_train, n_components = N_COMPONENTS):
        pca = RandomizedPCA(n_components, whiten=True).fit(X_train)
        eigenfaces = pca.components_.reshape((n_components, DEFAULT_FACE_SIZE, DEFAULT_FACE_SIZE))
        return pca, eigenfaces

    def transform_X(pca, X_train, X_test):
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca

    def train_model(pca, X_train_pca, y_train):
        print "\n======FITTING THE CLASSIFIER TO THE TRAINING SET======="
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        print("Done in {0:.3f} second(s).".format(time() - t0))
        print("Best estimator found by grid search: {0}".format(clf.best_estimator_))
        return clf

    def validate_model(clf, X_test_pca, y_test):
        y_pred = clf.predict(X_test_pca)
        return y_pred

    def print_comparison_result(y_test, y_predict):
        # Re-align numbering for result printing
        y_test = np.array(y_test)
        y_test = y_test - 1
        y_predict = y_predict - 1

        cm = confusion_matrix(y_test, y_predict, labels=range(3))
        df = pd.DataFrame(cm, columns = TARGET_NAMES, index = TARGET_NAMES)
        print "\n==================RESULT=================="
        print "Confusion Matrix: "
        print df
        print "Classification Report: "

        print(classification_report(y_test, y_predict, target_names=TARGET_NAMES))
        print """
        How to comprehend the report:
        - Recall value: "Given a true face of person X, how likely does the classifier detect it is X?
        - Precision value: "If the classifier predicted a face person X, how likely is it to be correct?
        """


        # print "SUMMARY:"
        # print "- Accuracy Score: {0:.2f}".format(accuracy_score(y_predict, y_test))
        # print "======TEST RESULT COMPARISON======="
        # for i, y in enumerate(y_test):
        #     print "{0}.\tPrediction = {1}\tActual/Predict: {2}/{3}".format(i,
        #         "CORRECT" if y == y_predict[i] else "INCORRECT",
        #         TARGET_NAMES[y-1], TARGET_NAMES[y_predict[i]-1])

    X_train, y_train = read_cropped_faces(mode=TRAINING)
    X_test, y_test = read_cropped_faces(mode=TESTING)

    pca, eigenfaces = compute_pca(X_train, y_train)
    X_train_pca, X_test_pca = transform_X(pca, X_train, X_test)
    clf = train_model(pca, X_train_pca, y_train)
    y_predict = validate_model(clf, X_test_pca, y_test)
    print_comparison_result(y_test, y_predict)



recognize_face()
