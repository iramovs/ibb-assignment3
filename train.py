import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import glob

def VGGNetModified(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    #model.add(Activation("softmax"))
    model.add(Activation("sigmoid"))

    return model


# parameters
epochs = 150
batch_size = 64
img_dims = (96,96,3)

# load train and test data
split_dir = "./database/split/"
split_data = np.load(split_dir+"split_data.npz")
trainData = split_data["trainData"]
testData = split_data["testData"]
trainLabels = split_data["trainLabels"]
testLabels = split_data["testLabels"]

# augmenting datset 
aug = ImageDataGenerator(rotation_range=25, 
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=False,
                         #brightness_range= [0.7, 1.3],
                         fill_mode="nearest")


# build model
model = VGGNetModified(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=100)

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# model training
H = model.fit_generator(aug.flow(trainData, trainLabels, batch_size=batch_size),
                        validation_data=(testData,testLabels),
                        steps_per_epoch=len(trainData) // batch_size,
                        epochs=epochs,
                        verbose=1)

# save the model to disk
model.save("./pretrained/VGGNetModified-%depochs" %(epochs))

# plot model training/validation accuracy while learning
plt.rcParams["font.family"] = "Times New Roman"

plt.figure()
plt.plot(np.arange(0,epochs), H.history["loss"], label="Train Loss Function")
plt.plot(np.arange(0,epochs), H.history["accuracy"], label="Train Accuracy")
plt.plot(np.arange(0,epochs), H.history["val_accuracy"], label="Test Accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.xlim([0, epochs])
plt.ylim([0, 1])
plt.legend()
plt.grid(True)

# save plot to disk
plt.savefig("./figures/train-loss-accuracy.svg")