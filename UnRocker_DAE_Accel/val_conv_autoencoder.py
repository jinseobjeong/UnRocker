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

session = tf.compat.v1.Session(config=config)


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
LENGTH = 256

(valX, valY) = ConvAutoencoder_1D.file_load_eval()
(input_min, input_max, input_median, data_min, data_max, data_median) = ConvAutoencoder_1D.file_data_range()
data_range = (data_max - data_min)
input_range = (input_max - input_min)

valX = np.expand_dims(valX, axis=-1)
valY = np.expand_dims(valY, axis=-1)

print("[INFO] making predictions...")


##
(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D.build(LENGTH)
eval_model.load_weights('./checkpoints_SOLO_x/my_checkpoint')
decoded2 = eval_model.predict(valX)
eval_model.save('./model2/model.h5')
eval_model.save('./model2/', save_format='tf')
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.figure()
	plt.style.use("ggplot")
	plt.plot(N, valX[index_n], linewidth = 2, color='red')
	plt.plot(N, valY[index_n], linewidth = 2, color='green')
	plt.plot(N, decoded2[index_n], linewidth = 2, color='blue')
	plt.title("Original & recovered data")
	plt.xlabel("Times")
	plt.ylabel("Value")
	plt.ylim((-0.1,1.1))
	plt.legend(loc="lower left")
	graph = "./eval_results/graph_eval%d" %i
	plt.savefig(graph)



plt.figure()
for i in range(0, 25):
	N = np.arange(0, LENGTH)
	index_n = i
	plt.subplot(5,5,i+1)
	plt.plot(N, valX[index_n], label="input_data", color='red')
	plt.plot(N, valY[index_n], label="true_data", color='green')
	plt.plot(N, decoded2[index_n], label="decoded_data",color='blue')
	plt.ylim((-1,2))
graph = "./eval_results/graph_sum_eval"
plt.savefig(graph)


