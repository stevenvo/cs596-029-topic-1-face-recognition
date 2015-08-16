# from pudb import set_trace

import os
import cv2
import sys
import re
import numpy as np
import pandas as pd
import pickle
import datetime
import config
import os.path


from glob import glob
from time import time
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

########################
#### CONFIGURATIONS ####
########################

# indicates all directories storing the cropped images of all 3 classes
CROPPED_IMAGES_DIRECTORY = '../data/{0}/*-Cropped'

# indicates all classes' name (label)
TARGET_NAMES = ['Tracy', 'Trish', 'Steven']

# n-component used to compute PCA
N_COMPONENTS = 30

# snapshot file (format) for storing the trained models
OUTPUT_TRAINED_MODEL_SNAPSHOT_FILE = "../data/trained_models/trained_models_{}.txt"
# indicate the filename where the the trained model can be loaded from
REUSE_TRAINED_MODEL_FILE = "../data/trained_models/trained_models_20150815_170111.txt" #leave this empty string if want to recompute the model

# dimension of the cropped faces
# IMPORTANT: should this value be changed, make sure you delete and regenerate all training and testing cropped images, otherwise the PCA will complain that the dimension of the data is not consistent. Also make sure the same dimension is used in the process_images.py.
DEFAULT_FACE_SIZE = 120.0

# CONSTANTS-DO-NOT-CHANGE
TRAINING = 'training'
TESTING = 'testing'



###################
#### FUNCTIONS ####
###################

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
        print "\n======{0} DATA=======".format(mode.upper())
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
        
    def snapshot_train_models(clf, pca):
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            'pca': pca, 
            'clf': clf
        }
        pickle.dump(data, open(OUTPUT_TRAINED_MODEL_SNAPSHOT_FILE.format(current_datetime), "wb"))
    
    def load_train_models(fname):
        try:
            data = pickle.load(open(fname, "rb"))
            return data
        except:
            e = sys.exc_info()[0]
            print "WARNING: Unable to load trained model file, program will recompute the model."
            return None

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

    
    trained_models = load_train_models(REUSE_TRAINED_MODEL_FILE)

    if trained_models == None: # Recompute models
        X_train, y_train = read_cropped_faces(mode=TRAINING)
        X_test, y_test = read_cropped_faces(mode=TESTING)
        pca, eigenfaces = compute_pca(X_train, y_train)
        X_train_pca, X_test_pca = transform_X(pca, X_train, X_test)
        clf = train_model(pca, X_train_pca, y_train)        
        snapshot_train_models(clf, pca) # store a snapshot of classifier for future re-usable

    else: # reuse trained models
        pca = trained_models["pca"] # load pca from saved trained model
        clf = trained_models["clf"] # load clf from saved trained model
        
        # Have to reload Training Data as well so the transform_X function can
        # compute X_text_pca in the "equivalent" range with X_train_pca
        X_train, y_train = read_cropped_faces(mode=TRAINING)
        X_test, y_test = read_cropped_faces(mode=TESTING)
        X_train_pca, X_test_pca = transform_X(pca, X_train, X_test)
        

    y_predict = validate_model(clf, X_test_pca, y_test)
    print_comparison_result(y_test, y_predict)


###################
# MAIN PROGRAM ####
###################

recognize_face()
