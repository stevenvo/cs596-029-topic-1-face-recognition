import numpy as np
import sys
import time
from scipy import ndimage
from skimage import feature
from skimage import io
from skimage import transform
from os import listdir
from os.path import isfile, join

orientations = 9
cell = 16
block = 3

if len(sys.argv) == 3:
    directory = sys.argv[1]
    vector_type = int(sys.argv[2])
    
    if vector_type not in [-1,1]:
        print "Invalid svm-value"
        sys.exit(0)

    filenames = [ f for f in listdir(directory + "/originals/") if isfile(join(directory + "/originals/",f)) and f[0] != "." ]
    print "Generating HOGs from '" + directory + "' folder:", len(filenames)
   
    percentages = {}
    for x in range(0, 100, 5):
        percentages[x] = True

    start_time = time.time()
    counter = 1
    for filename in filenames:

        val = int(float(counter) / len(filenames) * 100)
        if val in percentages and percentages[val]:
            print " Progress: %i %s" %(val, "%")
            percentages[val] = False

        print directory + "/originals/" + filename

        try:
            img = io.imread(directory + "/originals/" + filename, as_grey=True)
            
            vector = feature.hog(img, orientations=orientations, pixels_per_cell=(cell, cell), cells_per_block=(block, block), normalise=True)
            dimension = len(vector)
            vector = " ".join(map(str, vector.tolist()))
            
            filename = ".".join(filename.split(".")[:-1])
            temp = open(directory + "/vectors/" + filename + ".txt", 'w')
            temp.write(str(vector_type) + "\n" + str(dimension) + "\n" + vector)
            temp.close()
            counter += 1
        except Exception as e:
            print "ERROR:", e
            break

    total_time = time.time() - start_time
    print "Time: %.2f sec (Average: %.2f sec)" %(total_time, total_time / (counter-1)) # Prevent divide by zero
else:
    print "Usage: python", __file__, "directory svm-value[-1 (negative) | 1 (positive)]"



    