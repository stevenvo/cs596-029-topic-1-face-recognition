import os
import cv2
import sys

from glob import glob


LABEL = 'Trish'
ID = 2
DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING = '../data/training/{0:02}-{1}-Raw'.format(ID, LABEL)
DIRECTORY_OF_CROPPED_FACES = '../data/training/{0:02}-{1}-Cropped'.format(ID, LABEL)
DIRECTORY_OF_EIGEN_FACES = '../data/training/{0:02}-{1}-Eigen'.format(ID, LABEL)
CASCADE_FILE = './haarcascade_frontalface_default.xml'

def init():
    print "DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING={0}".format(DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING)
    print "DIRECTORY_OF_CROPPED_FACES={0}".format(DIRECTORY_OF_CROPPED_FACES)
    print "DIRECTORY_OF_EIGEN_FACES={0}".format(DIRECTORY_OF_EIGEN_FACES)


def create_directory():
    if not os.path.exists(DIRECTORY_OF_CROPPED_FACES):
        os.makedirs(DIRECTORY_OF_CROPPED_FACES)
    if not os.path.exists(DIRECTORY_OF_EIGEN_FACES):
        os.makedirs(DIRECTORY_OF_EIGEN_FACES)

def resize_image(img, new_wide=90.0, interpolation_method=cv2.INTER_AREA):
    ratio = float(new_wide) / img.shape[1]
    if ratio < 1: #only shrink, NOT enlarge
        new_dimension = (int(new_wide), int(img.shape[0] * ratio))
        return cv2.resize(img, new_dimension, interpolation= interpolation_method)
    else:
        return img


## TODO ##
#Each image is a 250x250 jpg, detected and centered using the openCV
#implementation of Viola-Jones face detector.  The cropping region
#returned by the detector was then automatically enlarged by a factor
#of 2.2 in each dimension to capture more of the head and then scaled
#to a uniform size.


def extract_faces_from_raw_images():

    def save_faces(directory, faces):
        for i, face in enumerate(faces):
            cv2.imwrite("{0}/{1}-{2}-face-{3:02}.jpg".format(directory, ID, LABEL, i), face)

    def draw_rectangle_on_detected_faces(image, faces):
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imshow("Faces found", image)
        return cv2.waitKey(0) # return key pressed

    def get_cropped_image(image, boxes):
        result = []
        for (x, y, w, h) in boxes:
            cropped_img = image[y:y+h, x:x+w]
            result.append(cropped_img)
        return result

    def preprocess_faces(faces):
        result = []
        for face in faces:
            face = resize_image(face, 120)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)
            result.append(face)
        return result


    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(CASCADE_FILE)
    files = glob("{0}/*.jpg".format(DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING))
    files.extend(glob("{0}/*.png".format(DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING)))
    files.extend(glob("{0}/*.bmp".format(DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING)))
    faces = []
    for fname in files:
        image = resize_image(cv2.imread(fname), 320)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_boxes = faceCascade.detectMultiScale(
            gray_img,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(15, 15),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        print "Found {0} faces in {1}!".format(len(face_boxes), fname)
        key_pressed = draw_rectangle_on_detected_faces(image.copy(), face_boxes)
        if key_pressed == ord('s'):
            cropped_faces = get_cropped_image(image, face_boxes)
            cropped_faces = preprocess_faces(cropped_faces)
            faces.extend(cropped_faces)
        elif key_pressed == ord('d'):
            os.remove(fname)

    save_faces(DIRECTORY_OF_CROPPED_FACES, faces)


###################
# MAIN PROGRAM ####
###################
init()
create_directory()
extract_faces_from_raw_images()
