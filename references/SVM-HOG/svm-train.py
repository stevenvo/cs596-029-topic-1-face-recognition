import svmlight
import numpy as np
from os import listdir
from os.path import isfile, join

training_data = []
directories = ["positives", "negatives"]

for directory in directories:

	directory += "/vectors/"
	filenames = [ f for f in listdir(directory) if isfile(join(directory,f)) and f[0] != "." ]
	print "Importing from '" + directory + "' folder:", len(filenames)

	percentages = {}
	for x in range(0, 100, 5):
		percentages[x] = True

	counter = 1	
	for filename in filenames:

		val = int(float(counter) / len(filenames) * 100)
		if val in percentages and percentages[val]:
			print " Progress: %i %s" %(val, "%")
			percentages[val] = False

		try:
			source = open(directory + filename, 'r')
			train_type = int(source.readline())
			train_num_dimensions = int(source.readline())
			train_dimensions = source.readline().strip().split()
			source.close()

		   	num = 1
		   	vals=[]
			for val in train_dimensions:
				vals.append((num, float(val)))
				num += 1
			
			training_data.append((train_type, vals))

		except Exception as e:
			print "ERROR:", e
			break
		counter += 1

print "Imported:", len(training_data), "\n"
print "Building Model"
model = svmlight.learn(training_data, type='classification', verbosity=0)
print "Write Model"
svmlight.write_model(model, 'svm-model.dat')

