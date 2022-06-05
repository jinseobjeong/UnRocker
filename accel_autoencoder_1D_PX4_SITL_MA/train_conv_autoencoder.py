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
LENGTH = 256#512#1024#128
# load the MNIST dataset
#print("[INFO] loading MNIST dataset...")
#((trainX, _), (testX, _)) = mnist.load_data()
# replace this load function to file open
# 


# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
#trainX = np.expand_dims(trainX, axis=-1)
#testX = np.expand_dims(testX, axis=-1)


#trainX = trainX.astype("float32") / 255.0
#testX = testX.astype("float32") / 255.0

(trainX, trainY, testX, testY, valX, valY) = ConvAutoencoder_1D.file_load_all()

trainX = np.expand_dims(trainX, axis=-1)
trainY = np.expand_dims(trainY, axis=-1)
testX = np.expand_dims(testX, axis=-1)
testY = np.expand_dims(testY, axis=-1)
valX = np.expand_dims(valX, axis=-1)
valY = np.expand_dims(valY, axis=-1)

#trainX = np.expand_dims(trainX, axis=-1)
#trainY = np.expand_dims(trainY, axis=-1)
#testX = np.expand_dims(testX, axis=-1)
#testY = np.expand_dims(testY, axis=-1)


# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder_1D.build(LENGTH)#10x10 insteadof 28x28
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

#plt.figure()

for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	#subplot_id = 661+i
	plt.figure()#subplot(5,5,i+1)
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
	plt.savefig(graph)#args["graph"])# %i])
#graph = "results/graph_sum"
#plt.savefig(graph)#args["graph"])# %i])

#	plt.savefig(args["plot%d" %i])

plt.figure()
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.subplot(5,5,i+1)
#	plt.style.use("ggplot")
	plt.plot(N, testX[index_n], label="input_data")
	plt.plot(N, testY[index_n], label="true_data")
	plt.plot(N, decoded[index_n], label="decoded_data")
	plt.ylim((-1,2))
graph = "results/graph_sum_test"
plt.savefig(graph)#args["graph"])# %i])


##
(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D.build(LENGTH)#10x10 insteadof 28x28
eval_model.load_weights('./checkpoints/my_checkpoint')
decoded2 = eval_model.predict(valX)

plt.figure()
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.subplot(5,5,i+1)
#	plt.style.use("ggplot")
	plt.plot(N, valX[index_n], label="input_data")
	plt.plot(N, valY[index_n], label="true_data")
	plt.plot(N, decoded2[index_n], label="decoded_data")
	plt.ylim((-1,2))
graph = "results/graph_sum_eval"
plt.savefig(graph)#args["graph"])# %i])




# loop over our number of output samples
##for i in range(0, args["samples"]):
	# grab the original image and reconstructed image
##	original = (valX[i] * 255).astype("uint8")
##	recon = (decoded[i] * 255).astype("uint8")
##	benign = (valY[i] * 255).astype("uint8")

	# stack the original and reconstructed image side-by-side
##	output = np.hstack([original, recon, benign])

	# if the outputs array is empty, initialize it as the current
	# side-by-side image display
##	if outputs is None:
##		outputs = output

	# otherwise, vertically stack the outputs
##	else:
##		outputs = np.vstack([outputs, output])

# save the outputs image to disk
#graph = "results/output"
#plt.savefig(graph)

#cv2.imwrite(graph, outputs)
##cv2.imwrite(args["output"], outputs)
