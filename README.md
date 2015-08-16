# CS596-029 Project Topic 1 - Face Recognition


## Project Directory Structure
```
├── data # TRAING & TEST DATA, AND SNAPSHOT OF TRAINED MODELS
│   ├── testing # TEST DATA
│   │   ├── 00-friends-Cropped # UNCLASSIFIED IMAGES
│   │   ├── 00-friends-Raw # UNCLASSIFIED IMAGES
│   │   ├── 01-Tracy-Cropped # CLASS 1 TEST DATA
│   │   ├── 02-Trish-Cropped # CLASS 2 TEST DATA
│   │   └── 03-Steven-Cropped # CLASS 3 TEST DATA
│   ├── trained_models
│   └── training # TRAINING DATA
│       ├── 01-Tracy-Cropped # CLASS 1 CROPPED FACES
│       ├── 01-Tracy-Raw # CLASS 1 RAW DATA
│       ├── 02-Trish-Cropped # CLASS 2 CROPPED FACES
│       ├── 02-Trish-Raw # CLASS 1 RAW DATA
│       ├── 03-Steven-Cropped # CLASS 3 CROPPED FACES
│       └── 03-Steven-Raw # CLASS 1 RAW DATA
├── docs # PROJECT POWERPOINT, DEVELOPMENT LOGS
├── references # CODE OR DOCUMENTS USED AS REFERENCES
└── src # PROGRAM SOURCE CODE

```

## Guide
Please make sure **you are in `src` folder** before executing any python command. 

The program has 2 python scripts:

* `process_images.py`: This script does all image processing tasks such as face extraction (Black and White), squaring, flipping and equalization. It reads images from **Raw** folders (refer to Directory Structure), then extract the cropped face outputs into **Cropped** folder.
    * **Configurations**: all parameters are configured and included inside the python file. You can change ID, Label to process more training data for different class (person). Make sure they are consistent. 
    * **Usage**: `python process_images.py`, when the image is popup, please click on the face you want to extract and press 's' to save it. If at any point you want to quit the process, press 'q'. Your last image processing progress will be kept in the *.npy files in RAW image folder. 
* `recognize_faces.py`: This script is the core data learning & validation. It loads the training data, compute PCA, generate eigenfaces and classifier, snapshot the classifier and PCA object, and predict results of the test data. 
    * **Configurations**: all parameters are configured and included inside the python file. You can play around with parameter `REUSE_TRAINED_MODEL_FILE`, change it to empty string if you want to recompute the model. 
    * **Usage**: `python recognize_faces.py` 

## Progress & Features

* **Completed features**
    1. [x] Using OpenCV to detect face    1. [x] Crop face and pre-process faces    1. [x] Preparing database for training & testing
    1. [x] Generate more training data by [Facebook Data Scraping](https://github.com/stevenvo/facebook_data_scraping)    1. [x] Face Flipping: to produce 2 images of the same person (for better recognition of both sides)    1. [x] Ability to click on the square box in OpenCV to choose which face can be extracted (since OpenCV is not always correct)    1. [x] Ability to load and store the progress of face extraction & review so we can resume later with newer images    1. [x] Implement GridSearchCV, kernel RBF to find best settings in multiple C and Gamma values.    1. [x] Add auto-scaling when displaying & processing large images.    1. [x] Printing Confusion Matrix in result (using SKLearn)Printing Classification Report in result (using SKLearn)    1. [x] Store and Load trained model using Pickle * **TODO List**    1. [ ] Applying adaboost, either with SVM or other classifier and compare the result. [experiment]    1. [ ] Build some UI function to show "Name" directly on the face in TEST photo. [fancy feature]    1. [ ] Find a way to deal with face angle (when it’s not straight) to improve result further.