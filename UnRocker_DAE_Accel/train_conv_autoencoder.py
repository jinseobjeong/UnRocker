# USAGE
# python train_conv_autoencoder.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.convautoencoder_1D import ConvAutoencoder_1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8,
	help="# number of samples to visualize when decoding")
ap.add_argument("-o", "--output", type=str, default="output.png",
	help="path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")
args = vars(ap.parse_args())

# initialize the number of epochs to train for and batch size
EPOCHS = 500
BS = 32
LENGTH = 256

(trainX, trainY, testX, testY, valX, valY) = ConvAutoencoder_1D.file_load_all()

trainX = np.expand_dims(trainX, axis=-1)
trainY = np.expand_dims(trainY, axis=-1)
testX = np.expand_dims(testX, axis=-1)
testY = np.expand_dims(testY, axis=-1)
valX = np.expand_dims(valX, axis=-1)
valY = np.expand_dims(valY, axis=-1)


# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder_1D.build(LENGTH)
opt = Adam(lr=1e-3)
autoencoder.compile(loss="mse", optimizer=opt)

# train the convolutional autoencoder
H = autoencoder.fit(
	trainX, trainY,
	validation_data=(testX, testY),
	epochs=EPOCHS,
	batch_size=BS)
#Save trained weights
autoencoder.save_weights('./checkpoints/my_checkpoint')

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

graph = "results/plot"
plt.savefig(graph)

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
outputs = None


for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.figure()
	plt.style.use("ggplot")
	plt.plot(N, testX[index_n], label="input_data")
	plt.plot(N, testY[index_n], label="true_data")
	plt.plot(N, decoded[index_n], label="decoded_data")
	plt.title("Original & recovered data")
	plt.xlabel("Times")
	plt.ylabel("Value")
	plt.ylim((-1,2))
	plt.legend(loc="lower left")
	graph = "results/graph%d" %i
	plt.savefig(graph)

plt.figure()
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.subplot(5,5,i+1)
	plt.plot(N, testX[index_n], label="input_data")
	plt.plot(N, testY[index_n], label="true_data")
	plt.plot(N, decoded[index_n], label="decoded_data")
	plt.ylim((-1,2))
graph = "results/graph_sum_test"
plt.savefig(graph)


(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D.build(LENGTH)
eval_model.load_weights('./checkpoints/my_checkpoint')
decoded2 = eval_model.predict(valX)

plt.figure()
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.subplot(5,5,i+1)
	plt.plot(N, valX[index_n], label="input_data")
	plt.plot(N, valY[index_n], label="true_data")
	plt.plot(N, decoded2[index_n], label="decoded_data")
	plt.ylim((-1,2))
graph = "results/graph_sum_eval"
plt.savefig(graph)


