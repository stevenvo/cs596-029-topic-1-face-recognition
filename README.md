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

* `process_images.py`: 
    * **Description**: This script read images from **Raw** folders (refer to Directory Structure), then extract the cropped faces into **Cropped** folder.    
    * **Parameters**: all parameters are included inside the python file. 
    * **Usage**: `python process_images.py`. 
* `recognize_faces.py`    

Process images