import os
import shutil
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array

src_dir = "./database/awe/"
dst_dir = "./database/prep/"
img_dims = (96,96,1)


# make prep directory for preprocessed images
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir)

# preprocess images
data = []
labels = []

for d in os.listdir(src_dir):
	if os.path.isdir(src_dir+d):
		
		label = d[1:]
		os.makedirs(dst_dir+label)
		
		f = open(src_dir+d+"/annotations.json")
		annot = json.load(f)

		for i in annot["data"]:

			nsrc = src_dir+d+"/"+annot["data"][i]["file"]
			ndst = dst_dir+label+"/"+annot["data"][i]["file"]

			image = cv2.imread(nsrc)
			image = cv2.resize(image, (img_dims[0],img_dims[1]))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			if annot["data"][i]["d"] == "l":
				image = cv2.flip(image, 1)
			
			# save images and labels to array and file
			cv2.imwrite(ndst, image)

			image = cv2.imread(ndst)
			image = img_to_array(image)
			data.append(image)

			labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42, stratify=labels)
trainLabels = to_categorical(trainLabels, num_classes=100)
testLabels = to_categorical(testLabels, num_classes=100)

# make split directory for splitted images
split_dir = "./database/split/"
if os.path.exists(split_dir):
    shutil.rmtree(split_dir)
os.makedirs(split_dir)

np.savez_compressed(split_dir+"split_data.npz",
                    trainData=trainData,
					testData=testData,
					trainLabels=trainLabels,
					testLabels=testLabels)