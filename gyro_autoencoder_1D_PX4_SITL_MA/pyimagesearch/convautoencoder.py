# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

class ConvAutoencoder:
	@staticmethod
	def build(width, height, depth, filters=(32, 64), latentDim=16):
		# initialize the input shape to be "channels last" along with
		# the channels dimension itself
		# channels dimension itself
		inputShape = (height, width, depth)
		chanDim = -1

		# define the input to the encoder
		inputs = Input(shape=inputShape)
		x = inputs

		# loop over the number of filters
		for f in filters:
			# apply a CONV => RELU => BN operation
			x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)

		# flatten the network and then construct our latent vector
		volumeSize = K.int_shape(x)
		x = Flatten()(x)
		latent = Dense(latentDim)(x)

		# build the encoder model
		encoder = Model(inputs, latent, name="encoder")

		# start building the decoder model which will accept the
		# output of the encoder as its inputs
		latentInputs = Input(shape=(latentDim,))
		x = Dense(np.prod(volumeSize[1:]))(latentInputs)
		x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

		# loop over our number of filters again, but this time in
		# reverse order
		for f in filters[::-1]:
			# apply a CONV_TRANSPOSE => RELU => BN operation
			x = Conv2DTranspose(f, (3, 3), strides=2,
				padding="same")(x)
			x = LeakyReLU(alpha=0.2)(x)
			x = BatchNormalization(axis=chanDim)(x)

		# apply a single CONV_TRANSPOSE layer used to recover the
		# original depth of the image
		x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
		outputs = Activation("sigmoid")(x)

		# build the decoder model
		decoder = Model(latentInputs, outputs, name="decoder")

		# our autoencoder is the encoder + decoder
		autoencoder = Model(inputs, decoder(encoder(inputs)),
			name="autoencoder")

		# return a 3-tuple of the encoder, decoder, and autoencoder
		return (encoder, decoder, autoencoder)

 
	def file_load_train():
		train_data = np.genfromtxt("dataset/RNN_8030Hz_merge_train.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		train_attack = train_data[:,5]
		train_gyro = train_data[:,2]
		data_h = train_data.shape[0]
		train_h = int(data_h-784)
		if train_h > 4000:
			train_h = 4000

		train_input = np.empty((train_h,28,28))
		train_output = np.empty((train_h,28,28))

		for index in range(train_h):
			for sub_index in range(28):
				train_input[index][sub_index] = train_attack[index+sub_index*28:index+sub_index*28+28]
				train_output[index][sub_index] = train_gyro[index+sub_index*28:index+sub_index*28+28]
		train_input = train_input.astype("float32") / 5.0 + 1.0
		train_output = train_output.astype("float32") / 0.02 + 1.0
		return (train_input, train_output)

	def file_load_test():
		test_data = np.genfromtxt("dataset/RNN_8030Hz_merge_test.csv", delimiter=",", dtype=np.float32, skip_header=True)[:,1:]
		test_attack = test_data[:,5]
		test_gyro = test_data[:,2]
		data_h = test_data.shape[0]
		test_h = int(data_h-784)

		if test_h > 1000:
			test_h = 1000

		test_input = np.empty((test_h,28,28))
		test_output = np.empty((test_h,28,28))

		for index in range(test_h):
			for sub_index in range(28):
				test_input[index][sub_index] = test_attack[index+sub_index*28:index+sub_index*28+28]
				test_output[index][sub_index] = test_gyro[index+sub_index*28:index+sub_index*28+28]
		test_input = test_input.astype("float32") / 5.0 + 1.0
		test_output = test_output.astype("float32") / 0.02 + 1.0
		return (test_input, test_output)

