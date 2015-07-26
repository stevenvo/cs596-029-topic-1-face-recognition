from pudb import set_trace

import os
import cv2
import sys

from glob import glob

TYPE = 'testing'
LABEL = 'friends'
ID = 0
DIRECTORY_OF_RAW_IMAGES = '../data/{0}/{1:02}-{2}-Raw'.format(TYPE, ID, LABEL)
DIRECTORY_OF_CROPPED_FACES = '../data/{0}/{1:02}-{2}-Cropped'.format(TYPE, ID, LABEL)
DIRECTORY_OF_EIGEN_FACES = '../data/{0}/{1:02}-{2}-Eigen'.format(TYPE, ID, LABEL)
CASCADE_FILE = './haarcascade_frontalface_default.xml'
DEFAULT_FACE_SIZE = 120.0
count = 0

def init():
    print "DIRECTORY_OF_RAW_IMAGES={0}".format(DIRECTORY_OF_RAW_IMAGES)
    print "DIRECTORY_OF_CROPPED_FACES={0}".format(DIRECTORY_OF_CROPPED_FACES)
    print "DIRECTORY_OF_EIGEN_FACES={0}".format(DIRECTORY_OF_EIGEN_FACES)

def create_directory():
    if not os.path.exists(DIRECTORY_OF_CROPPED_FACES):
        os.makedirs(DIRECTORY_OF_CROPPED_FACES)
    if not os.path.exists(DIRECTORY_OF_EIGEN_FACES):
        os.makedirs(DIRECTORY_OF_EIGEN_FACES)

def resize_square_image(img, new_wide=120.0, can_enlarge=False):
    ratio = float(new_wide) / img.shape[1]
    if ratio < 1: #shrink
        print "resize_square_image:shrink"
        new_dimension = (int(new_wide), int(new_wide))
        return cv2.resize(img, new_dimension, interpolation=cv2.INTER_AREA)
    elif can_enlarge: #enlarge if allowed
        print "resize_square_image:enlarge"
        new_dimension = (int(new_wide), int(new_wide))
        return cv2.resize(img, new_dimension, interpolation=cv2.INTER_LINEAR)
    else: #no enlarge
        print "resize_square_image:as-is"
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

    # - find the ratio from face_box to default_cropped_size=[100.0, 100.0]
	#    ratio r = default_cropped_size[0] / face_box.height
    # - crop the image + PADDING
    # - scale the cropped image by ratio r

    # base on the center, convert rectangle to square
    def convert_box_to_square(x, y, w, h):
        if h > w:
            x = x + w/2 - h/2
            w = h
        elif h < w:
            y = y + h/2 - w/2
            h = w
        return x, y, w, h

    # r must be smaller than 2, for example 1.2
    def add_padding_to_square_box(x, y, w, h, r=1.2):
        padding = 1
        while (x-padding > 1) and (y-padding > 1) and (x+padding < w) and (y+padding <h) and padding < w * (r-1):
            padding += 1
        return x-padding, y-padding, w+padding, h+padding


    def get_cropped_image(image, boxes, padding_ratio=1.2, default_cropped_size=120.0):
        result = []
        for (x, y, w, h) in boxes:
            # global count
            # if count == 17:
                # set_trace()
            # print "ORG x, y, w, h = {0}, {1}, {2}, {3}".format(x, y, w, h)
            x, y, w, h = convert_box_to_square(x, y, w, h) # "squaren" the face box
            # print "SQR x, y, w, h = {0}, {1}, {2}, {3}".format(x, y, w, h)
            # x, y, w, h = add_padding_to_square_box(x, y, w, h, padding_ratio) #add padding to the face box
            # print "PAD x, y, w, h = {0}, {1}, {2}, {3}".format(x, y, w, h)
            cropped_img = image[y:y+h, x:x+w]
            # print cropped_img.shape
            resized_img = resize_square_image(cropped_img, new_wide=default_cropped_size,
                can_enlarge=True)
            # print resized_img.shape
            result.append(resized_img)
            # count += 1
        return result

    def preprocess_faces(faces):
        result = []
        for face in faces:
            # face = resize_square_image(face, 120)
            face = cv2.equalizeHist(face)
            result.append(face)
        return result


    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(CASCADE_FILE)
    files = glob("{0}/*.jpg".format(DIRECTORY_OF_RAW_IMAGES))
    files.extend(glob("{0}/*.png".format(DIRECTORY_OF_RAW_IMAGES)))
    files.extend(glob("{0}/*.bmp".format(DIRECTORY_OF_RAW_IMAGES)))
    faces = []
    for fname in files:
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        face_boxes = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(15, 15),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        print "Found {0} faces in {1}!".format(len(face_boxes), fname)
        key_pressed = draw_rectangle_on_detected_faces(image.copy(), face_boxes)
        if key_pressed == ord('s'):
            cropped_faces = get_cropped_image(image, face_boxes,
                padding_ratio=1.2, default_cropped_size=DEFAULT_FACE_SIZE)
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
