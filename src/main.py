import os
import glob
import cv2
import sys
import ConfigParser #reading facebook consumer key and secret

#using 'facebook' module for oauth with FaceBook
# create a file called 'auth_settings.txt' in 'src' directory
# the file use below structure (remove # sign):
# [facebook]
# app_id = "<app_id>"
# app_secret = "<app_secret>"
import facebook


LABEL = 'Tracy'
ID = 1
FACEBOOK_PROFILE_ID = ''
DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING = '../data/input/training/{0:02}-{1}-Raw'.format(ID, LABEL)
DIRECTORY_OF_CROPPED_FACES = '../data/input/training/{0:02}-{1}-Cropped'.format(ID, LABEL)
DIRECTORY_OF_EIGEN_FACES = '../data/input/training/{0:02}-{1}-Eigen'.format(ID, LABEL)
CASCADE_FILE = './haarcascade_frontalface_default.xml'
CONFIG_FILE = './auth_settings.txt'
config = ConfigParser.RawConfigParser()

def init():
    config.read(CONFIG_FILE)

def create_directory():
    if not os.path.exists(DIRECTORY_OF_CROPPED_FACES):
        os.makedirs(DIRECTORY_OF_CROPPED_FACES)
    if not os.path.exists(DIRECTORY_OF_EIGEN_FACES):
        os.makedirs(DIRECTORY_OF_EIGEN_FACES)

def download_friends_photos():
    token = facebook.get_app_access_token(config.get('facebook', 'app_id'),
        config.get('facebook', 'app_secret'))
    print token

def extract_cropped_faces():

    def save_faces(directory, faces):
        for i, face in enumerate(faces):
            cv2.imwrite("{0}/{1}-{2}-face-{3:02}.jpg".format(directory, ID, LABEL, i), face)

    def draw_rectangle_on_detected_faces(image, faces):
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imshow("Faces found", image)
        return cv2.waitKey(0) # return key pressed

    def get_cropped_image(image, boxes, cropped_imgs):
        for (x, y, w, h) in boxes:
            cropped_img = image[y:y+h, x:x+w]
            cropped_imgs.append(cropped_img)

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(CASCADE_FILE)

    face_cropped_images = []

    # Iterate through all image files
    for fname in glob.glob("{0}/*.jpg".format(DIRECTORY_OF_RAW_IMAGES_FOR_TRAINING)):
        image = cv2.imread(fname)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_boxes = faceCascade.detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(15, 15),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        print "Found {0} faces in {1}!".format(len(face_boxes), fname)
        key_pressed = draw_rectangle_on_detected_faces(image.copy(), face_boxes)
        if key_pressed == ord('s'):
            get_cropped_image(image, face_boxes, face_cropped_images)
        elif key_pressed == ord('d'):
            os.remove(fname)

    save_faces(DIRECTORY_OF_CROPPED_FACES, face_cropped_images)

def preprocess_faces():
    def resize_image(img, new_wide=90.0, interpolation_method=cv2.INTER_AREA):
        ratio = new_wide / img.shape[1]
        new_dimension = (int(new_wide), int(img.shape[0] * ratio))
        return cv2.resize(img, new_dimension, interpolation= interpolation_method)

    for fname in glob.glob("{0}/*.jpg".format(DIRECTORY_OF_CROPPED_FACES)):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) #read image in GrayScale
        resized_img = resize_image(img)
        equalized_img = cv2.equalizeHist(resized_img)
        cv2.imshow("equalized img", equalized_img)
        cv2.waitKey(0)


###################
# MAIN PROGRAM#####
###################
init()
create_directory()
download_friends_photos()
# extract_cropped_faces()
# preprocess_faces()
