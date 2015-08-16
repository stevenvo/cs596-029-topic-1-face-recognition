# from pudb import set_trace

import os
import cv2
import sys
import uuid
import pickle
import numpy as np

from glob import glob

########################
#### CONFIGURATIONS ####
########################

# indicates which dataset the processed images should fall into, change to 'test' if you want to populate more photos for testing purposes.
TYPE = 'training'

# Currently the program support static number of classes (persons). These are the pre-defined classes.
# {ID:1, LABEL:Tracy}
# {ID:2, LABEL:Trish}
# {ID:3, LABEL:Steven}
# IMPORTANT: when modifying the configuration below, make sure the ID tallies with the LABEL.

# indicates which class (person) name & ID the processed images should be tagged with.
ID, LABEL = 2, 'Trish'

# indicates where the RAW files are placed, the program will grab the raw files from these directories.
DIRECTORY_OF_RAW_IMAGES = '../data/{0}/{1:02}-{2}-Raw'.format(TYPE, ID, LABEL)
# due to a massive amount of RAW files and to avoid duplicate processing, the progress.npy is used to mark which RAW files have been processed before. This file is stored in each RAW files directory, in numpy format. 
PROGRESS_FILE = '../data/{0}/{1:02}-{2}-Raw/progress.npy'.format(TYPE, ID, LABEL)
# indicates the directory storing the cropped images of the face after processing
DIRECTORY_OF_CROPPED_FACES = '../data/{0}/{1:02}-{2}-Cropped'.format(TYPE, ID, LABEL)
# cascade file path
CASCADE_FILE = './haarcascade_frontalface_default.xml'

# dimension of the cropped faces
# IMPORTANT: should this value be changed, make sure you delete and regenerate all training and testing cropped images, otherwise the PCA will complain that the dimension of the data is not consistent. Also make sure the same dimension is used in the recognize_face.py.
DEFAULT_FACE_SIZE = 120.0


###################
#### FUNCTIONS ####
###################

def init():
    print "DIRECTORY_OF_RAW_IMAGES={0}".format(DIRECTORY_OF_RAW_IMAGES)
    print "DIRECTORY_OF_CROPPED_FACES={0}".format(DIRECTORY_OF_CROPPED_FACES)
    # print "DIRECTORY_OF_EIGEN_FACES={0}".format(DIRECTORY_OF_EIGEN_FACES)

def create_directory():
    if not os.path.exists(DIRECTORY_OF_CROPPED_FACES):
        os.makedirs(DIRECTORY_OF_CROPPED_FACES)
    # if not os.path.exists(DIRECTORY_OF_EIGEN_FACES):
    #     os.makedirs(DIRECTORY_OF_EIGEN_FACES)

def resize_image_if_bigger(img, max_edge = 1024):
    (h,w) = img.shape[:2]
    if h > max_edge or img.shape[0] > max_edge:
        # print "resize_image_if_bigger is called!"
        l = max(h, w)
        ratio = float(max_edge) / l
        new_dimension = (int(w*ratio), int(h*ratio))
        return cv2.resize(img, new_dimension, interpolation=cv2.INTER_LINEAR)
    return img

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


def extract_faces_from_raw_images():

    def save_face(directory, face):
        f_path = "{0}/{1}-{2}-face-{3}.jpg".format(directory, ID, LABEL, uuid.uuid4())
        cv2.imwrite(f_path, face)
        return f_path

    def save_faces(directory, faces):
        for i, face in enumerate(faces):
            # cv2.imwrite("{0}/{1}-{2}-face-{3:02}.jpg".format(directory, ID, LABEL, i), face)
            cv2.imwrite("{0}/{1}-{2}-face-{3}.jpg".format(directory, ID, LABEL, uuid.uuid4()), face)


    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global clicked_pnt

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pnt = [x, y]
            # print refPt
            # cropping = True

        # check to see if the left mouse button was released
        # elif event == cv2.EVENT_LBUTTONUP:
        #     # record the ending (x, y) coordinates and indicate that
        #     # the cropping operation is finished
        #     refPt.append((x, y))
        #     print refPt
        #     cropping = False


    def draw_rectangle_on_detected_faces(image, faces):
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imshow("FacesFound", image)
        cv2.setMouseCallback("FacesFound", click_and_crop)
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


    def get_cropped_image(image, boxes, padding_ratio=1.2, default_cropped_size=120.0, ref_pnt = None):
        result = []
        for (x, y, w, h) in boxes:
            x, y, w, h = convert_box_to_square(x, y, w, h) # "squaren" the face box
            cropped_img = image[y:y+h, x:x+w]
            # print "x,y,w,h: {0}, {1}, {2}, {3}".format(x,y,w,h)
            # print "ref_pnt: {0}".format(ref_pnt)

            if ref_pnt != None and len(ref_pnt) == 2:
                if (x <= ref_pnt[0] <= x + w) and (y<= ref_pnt[1] <= y + h):
                    # print "OK - x,y,w,h: {0}, {1}, {2}, {3}".format(x,y,w,h)
                    resized_img = resize_square_image(cropped_img, new_wide=default_cropped_size,
                        can_enlarge=True)
                    result.append(resized_img)
                    return result
            else:
                resized_img = resize_square_image(cropped_img, new_wide=default_cropped_size,
                    can_enlarge=True)
                # result.append(resized_img)

        return result

    def preprocess_faces(faces):
        result = []
        for face in faces:
            # face = resize_square_image(face, 120)
            face = cv2.equalizeHist(face)
            result.append(face)
        return result

    def already_processed(progress_file_data, fname):
        for i, files in enumerate(progress_file_data):
            if files[0] == fname:
                return i
        return -1

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(CASCADE_FILE)
    files = glob("{0}/*.jpg".format(DIRECTORY_OF_RAW_IMAGES))
    files.extend(glob("{0}/*.png".format(DIRECTORY_OF_RAW_IMAGES)))
    files.extend(glob("{0}/*.bmp".format(DIRECTORY_OF_RAW_IMAGES)))
    faces = []

    if os.path.isfile(PROGRESS_FILE):
        progress_file_data = np.load(PROGRESS_FILE).tolist()
    else:
        progress_file_data = []
    # print progress_file_data
    for fname in files:
        if already_processed(progress_file_data, fname) == -1: #is NOT processed yet
            image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            image = resize_image_if_bigger(image, max_edge=1024)
            face_boxes = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(15, 15),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            print "Found {0} faces in {1}!".format(len(face_boxes), fname)
            global clicked_pnt

            # color image just for show
            color_image = cv2.imread(fname)
            color_image = resize_image_if_bigger(color_image, max_edge=1024)
            key_pressed = draw_rectangle_on_detected_faces(color_image, face_boxes)

            if key_pressed == ord('s'):
                cropped_faces = get_cropped_image(image, face_boxes,
                    padding_ratio=1.2, default_cropped_size=DEFAULT_FACE_SIZE, ref_pnt = clicked_pnt)
                cropped_faces = preprocess_faces(cropped_faces)
                cropped_face = cropped_faces[0]
                flipped_face = cv2.flip(cropped_face, 1)
                cropped_file_path = save_face(DIRECTORY_OF_CROPPED_FACES, cropped_face)
                save_face(DIRECTORY_OF_CROPPED_FACES, flipped_face)
                progress_file_data.append([fname, cropped_file_path])
                # faces.extend(cropped_faces)
            elif key_pressed == ord('d'):
                os.remove(fname)
            elif key_pressed == ord('q'):
                break

    np.save(PROGRESS_FILE, progress_file_data)

    # save_faces(DIRECTORY_OF_CROPPED_FACES, faces)


###################
# MAIN PROGRAM ####
###################
init()
create_directory()
extract_faces_from_raw_images()
