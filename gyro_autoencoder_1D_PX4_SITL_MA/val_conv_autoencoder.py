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
import time
from tensorflow.compat.v1.keras.experimental import export_saved_model 
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction =0.8

session = tf.compat.v1.Session(config=config)#InteractiveSession(config=config)


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
EPOCHS = 25
BS = 32
LENGTH = 256#512#256#1024#128

(valX, valY) = ConvAutoencoder_1D.file_load_eval()
(input_min, input_max, input_median, data_min, data_max, data_median) = ConvAutoencoder_1D.file_data_range()
data_range = (data_max - data_min)
input_range = (input_max - input_min)

valX = np.expand_dims(valX, axis=-1)
valY = np.expand_dims(valY, axis=-1)


print("[INFO] making predictions...")

##
(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D.build(LENGTH)#10x10 insteadof 28x28

eval_model.load_weights('./checkpoints_SOLO_GyrX2/my_checkpoint')
#eval_model.load_weights('./checkpoints_220304_solo_256/my_checkpoint')
#eval_model.load_weights('./checkpoints_210531_gyro_LUT_sitl_MA/my_checkpoint')
#eval_model.load_weights('./checkpoints_220128_gyro_sitl_LUT_delayed_compromised_mixed_MA_128_SOLO/my_checkpoint')
#eval_model.load_weights('./checkpoints_210612_gyro_sitl_LUT_delayed_compromised_mixed_MA_128/my_checkpoint')
decoded2 = eval_model.predict(valX)
eval_model.save('./model2/model.h5')
eval_model.save('./model2/', save_format='tf')
#export_saved_model(eval_model, './model2')
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	#subplot_id = 661+i
	plt.figure()#subplot(5,5,i+1)
	plt.style.use("ggplot")
	plt.plot(N, valX[index_n], label="input_data")
	plt.plot(N, valY[index_n], label="true_data")
	plt.plot(N, decoded2[index_n], label="decoded_data")
	plt.title("Original & recovered data")
	plt.xlabel("Times")
	plt.ylabel("Value")
	plt.ylim((-1,2))
	plt.legend(loc="lower left")
	graph = "./eval_results/graph_eval%d" %i
	plt.savefig(graph)#args["graph"])# %i])



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
graph = "./eval_results/graph_sum_eval"
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
