from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, top_k_accuracy_score
from sklearn.dummy import DummyClassifier
import random

def print_pred():
	correct = 0
	for i in range(200):
		print("%d recognised as %d with %.2f" %(np.argmax(testLabels[i]), np.argmax(predLabels[i]), np.max(predLabels[i])))
		if np.argmax(testLabels[i]) == np.argmax(predLabels[i]):
			correct += 1
	print("Correct: %d/%d" %(correct, 200))

def cmc_scores(predLabels):
	print(top_k_accuracy_score(testLabels.argmax(axis=1), predLabels, k=1)*100)
	return [top_k_accuracy_score(testLabels.argmax(axis=1), predLabels, k=i)*100 for i in range(1,100)]


# set pretty print for numpy
np.set_printoptions(precision=3, suppress=True)

# load test data
split_dir = "./database/split/"
split_data = np.load(split_dir+"split_data.npz")
trainData = split_data["trainData"]
testData = split_data["testData"]
trainLabels = split_data["trainLabels"]
testLabels = split_data["testLabels"]

# predict with random classifier
dummy_clf = DummyClassifier(strategy="uniform")
dummy_clf.fit(trainData, trainLabels)
predDummyLabels = dummy_clf.predict(testData)


# plot cmc_curve
plt.rcParams["font.family"] = "Times New Roman"

ranks = np.arange(1,100)

model = keras.models.load_model("./pretrained/VGGNetModified-20epochs")
predLabels = model.predict(testData)
plt.plot(ranks, cmc_scores(predLabels), label="VGGNet-modified(20 epoch)")

model = keras.models.load_model("./pretrained/VGGNetModified-40epochs")
predLabels = model.predict(testData)
plt.plot(ranks, cmc_scores(predLabels), label="VGGNet-modified(40 epoch)")

model = keras.models.load_model("./pretrained/VGGNetModified-60epochs")
predLabels = model.predict(testData)
plt.plot(ranks, cmc_scores(predLabels), label="VGGNet-modified (60 epoch)")

model = keras.models.load_model("./pretrained/VGGNetModified-150epochs")
predLabels = model.predict(testData)
plt.plot(ranks, cmc_scores(predLabels), label="VGGNet-modified (150 epoch)")

plt.plot(ranks, cmc_scores(predDummyLabels), label="Random")
plt.xlabel("Rank")
plt.xlim([0, 40])
plt.ylim([0, 100])
plt.ylabel("Accuracy (%)")
plt.title('CMC Curve')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("./figures/cmc-curve.svg")