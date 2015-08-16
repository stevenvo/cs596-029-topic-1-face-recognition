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

* `process_images.py`: reads images from **Raw** folders (refer to Directory Structure), then extract the cropped faces into **Cropped** folder.    
    * **Configurations**: all parameters are configured and included inside the python file. You can change ID, Label to process more training data for different class (person). Make sure they are consistent. 
    * **Usage**: `python process_images.py`, when the image popup, please click on the face you want to extract and press 's' to save it. If you want to quit, press 'q'. Your image processing progress will be kept in the *.npy files in RAW image folder. 
* `recognize_faces.py`: learns from the training data, generate models, snapshot the model, predict results. 
    * **Configurations**: all parameters are configured and included inside the python file. You can play around with `REUSE_TRAINED_MODEL_FILE`, change it to empty string if you want to recompute the model. 
    * **Usage**: `python recognize_faces.py`

Process images