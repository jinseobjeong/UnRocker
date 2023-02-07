# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import UpSampling1D

from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import numpy as np

class ConvAutoencoder_1D:
	@staticmethod
	def build(length, latentDim=64):
		inputShape = (length, 1)
		input_sig = Input(shape=inputShape)#, batch_shape=(1, 32,1))
		x = Conv1D(16,3, activation='relu', padding='same',dilation_rate=2)(input_sig)
		x1 = MaxPooling1D(2)(x)
		x2 = Conv1D(8,3, activation='relu', padding='same',dilation_rate=2)(x1)
		x3 = MaxPooling1D(2)(x2)
		x4 = AveragePooling1D()(x3)
		flat = Flatten()(x4)
		encoded = Dense(latentDim)(flat)
		encoder = Model(input_sig, encoded, name="encoder")
		
		latentInputs = Input(shape=(latentDim,))
		d1 = Dense(128)(latentInputs)
		d2 = Reshape((32,4))(d1)
		d3 = Conv1D(8,1,strides=1, activation='relu', padding='same')(d2)
		d4 = UpSampling1D(2)(d3)
		d5 = Conv1D(16,1,strides=1, activation='relu', padding='same')(d4)
		d6 = UpSampling1D(2)(d5)
		d7 = UpSampling1D(2)(d6)
		decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d7)
		decoder = Model(latentInputs, decoded, name="decoder")

		autoencoder = Model(input_sig, decoder(encoder(input_sig)),
			name="autoencoder")

		opt = Adam(lr=1e-3)
		autoencoder.compile(loss="mse", optimizer=opt)

		return (encoder, decoder, autoencoder)

	def file_load_all():
		a_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Training1_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h1 = a_data1.shape[0]
		a_w1 = a_data1.shape[1]
		a_data2 = np.genfromtxt("DNN_Training/DNN_PX4_Training2_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h2 = a_data2.shape[0]
		a_w2 = a_data2.shape[1]
		a_data3 = np.genfromtxt("DNN_Training/DNN_PX4_Training3_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h3 = a_data3.shape[0]
		a_w3 = a_data3.shape[1]
		a_data4 = np.genfromtxt("DNN_Training/DNN_PX4_Training4_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h4 = a_data4.shape[0]
		a_w4 = a_data4.shape[1]
		a_data5 = np.genfromtxt("DNN_Training/DNN_PX4_Training5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h5 = a_data5.shape[0]
		a_w5 = a_data5.shape[1]
		a_data6 = np.genfromtxt("DNN_Training/DNN_PX4_Training6_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h6 = a_data6.shape[0]
		a_w6 = a_data6.shape[1]
		a_data7 = np.genfromtxt("DNN_Training/DNN_PX4_Training7_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h7 = a_data7.shape[0]
		a_w7 = a_data7.shape[1]
		a_data8 = np.genfromtxt("DNN_Training/DNN_PX4_Training8_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h8 = a_data8.shape[0]
		a_w8 = a_data8.shape[1]
		a_data9 = np.genfromtxt("DNN_Training/DNN_PX4_Training9_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h9 = a_data9.shape[0]
		a_w9 = a_data9.shape[1]
		a_data10 = np.genfromtxt("DNN_Training/DNN_PX4_Training10_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h10 = a_data10.shape[0]
		a_w10 = a_data10.shape[1]
		a_data11 = np.genfromtxt("DNN_Training/DNN_PX4_Training11_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h11 = a_data11.shape[0]
		a_w11 = a_data11.shape[1]
		a_data12 = np.genfromtxt("DNN_Training/DNN_PX4_Training12_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h12 = a_data12.shape[0]
		a_w12 = a_data12.shape[1]
		a_data13 = np.genfromtxt("DNN_Training/DNN_PX4_Training13_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h13 = a_data13.shape[0]
		a_w13 = a_data13.shape[1]
		a_data14 = np.genfromtxt("DNN_Training/DNN_PX4_Training14_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h14 = a_data14.shape[0]
		a_w14 = a_data14.shape[1]
		a_data15 = np.genfromtxt("DNN_Training/DNN_PX4_Training15_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h15 = a_data15.shape[0]
		a_w15 = a_data15.shape[1]
		a_data16 = np.genfromtxt("DNN_Training/DNN_PX4_Training16_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h16 = a_data16.shape[0]
		a_w16 = a_data16.shape[1]
		a_data17 = np.genfromtxt("DNN_Training/DNN_PX4_Training17_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h17 = a_data17.shape[0]
		a_w17 = a_data17.shape[1]
		a_data18 = np.genfromtxt("DNN_Training/DNN_PX4_Training18_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h18 = a_data18.shape[0]
		a_w18 = a_data18.shape[1]
		a_data19 = np.genfromtxt("DNN_Training/DNN_PX4_Training19_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h19 = a_data19.shape[0]
		a_w19 = a_data19.shape[1]
		a_data20 = np.genfromtxt("DNN_Training/DNN_PX4_Training20_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h20 = a_data20.shape[0]
		a_w20 = a_data20.shape[1]

		
#		print('original csv size %d x %d' %(all_h, all_w))
		length = 256#1024#128
		step_size = 4#10
		dataset_h1 = int((a_h1-length)/step_size)
		dataset_h2 = int((a_h2-length)/step_size)
		dataset_h3 = int((a_h3-length)/step_size)
		dataset_h4 = int((a_h4-length)/step_size)
		dataset_h5 = int((a_h5-length)/step_size)
		dataset_h6 = int((a_h6-length)/step_size)
		dataset_h7 = int((a_h7-length)/step_size)
		dataset_h8 = int((a_h8-length)/step_size)
		dataset_h9 = int((a_h9-length)/step_size)
		dataset_h10 = int((a_h10-length)/step_size)
		dataset_h11 = int((a_h11-length)/step_size)
		dataset_h12 = int((a_h12-length)/step_size)
		dataset_h13 = int((a_h13-length)/step_size)
		dataset_h14 = int((a_h14-length)/step_size)
		dataset_h15 = int((a_h15-length)/step_size)
		dataset_h16 = int((a_h16-length)/step_size)
		dataset_h17 = int((a_h17-length)/step_size)
		dataset_h18 = int((a_h18-length)/step_size)
		dataset_h19 = int((a_h19-length)/step_size)
		dataset_h20 = int((a_h20-length)/step_size)

		dataset_h = dataset_h1+dataset_h2+dataset_h3+dataset_h4+dataset_h5+dataset_h6+dataset_h7+dataset_h8+dataset_h9+dataset_h10+dataset_h11+dataset_h12+dataset_h13+dataset_h14+dataset_h15+dataset_h16+dataset_h17+dataset_h18+dataset_h19+dataset_h20



		all_dataset = np.empty((dataset_h,length,2))
		index_temp = 0

		for index in range(dataset_h1):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data1[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data1[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h1

		for index in range(dataset_h2):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data2[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data2[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h2

		for index in range(dataset_h3):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data3[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data3[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h3

		for index in range(dataset_h4):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data4[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data4[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h4

		for index in range(dataset_h5):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data5[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data5[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h5

		for index in range(dataset_h6):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data6[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data6[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h6

		for index in range(dataset_h7):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data7[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data7[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h7

		for index in range(dataset_h8):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data8[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data8[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h8

		for index in range(dataset_h9):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data9[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data9[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h9

		for index in range(dataset_h10):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data10[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data10[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h10

		for index in range(dataset_h11):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data11[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data11[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h11

		for index in range(dataset_h12):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data12[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data12[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h12

		for index in range(dataset_h13):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data13[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data13[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h13

		for index in range(dataset_h14):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data14[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data14[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h14

		for index in range(dataset_h15):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data15[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data15[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h15

		for index in range(dataset_h16):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data16[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data16[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h16

		for index in range(dataset_h17):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data17[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data17[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h17

		for index in range(dataset_h18):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data18[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data18[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h18

		for index in range(dataset_h19):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data19[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data19[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h19

		for index in range(dataset_h20):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data20[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data20[index*step_size+sub_index][9]

		np.take(all_dataset, np.random.permutation(all_dataset.shape[0]),axis=0,out=all_dataset)

		all_attack = np.empty((dataset_h,length))
		all_gyro = np.empty((dataset_h,length))

		for index in range(dataset_h):
			for sub_index in range(length):
				all_attack[index][sub_index] = all_dataset[index][sub_index][0]
				all_gyro[index][sub_index] = all_dataset[index][sub_index][1]

		data_max = np.max(all_gyro)
		data_min = np.min(all_gyro)
		data_range = (data_max - data_min)
		data_median = (data_max + data_min)/2

		input_max = np.max(all_attack)
		input_min = np.min(all_attack)
		input_range = (input_max - input_min)
		input_median = (input_max + input_min)/2

		upper_bound = input_max - data_range
		lower_bound = input_min + data_range

		all_attack = (all_attack.astype("float32")-input_median) / input_range + 0.5
		all_gyro = (all_gyro.astype("float32")-data_median) / data_range + 0.5


		train_h = dataset_h

#		train_input = np.empty((train_h,length))
#		train_output = np.empty((train_h,length))
		
#		train_input = all_attack[0:train_h-1,:]
#		train_output = all_gyro[0:train_h-1,:]

		##test set processing

		at_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Eval1_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h1 = at_data1.shape[0]
		at_w1 = at_data1.shape[1]
		at_data2 = np.genfromtxt("DNN_Training/DNN_PX4_Eval2_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h2 = at_data2.shape[0]
		at_w2 = at_data2.shape[1]
		at_data3 = np.genfromtxt("DNN_Training/DNN_PX4_Eval3_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h3 = at_data3.shape[0]
		at_w3 = at_data3.shape[1]
		at_data4 = np.genfromtxt("DNN_Training/DNN_PX4_Eval4_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h4 = at_data4.shape[0]
		at_w4 = at_data4.shape[1]
		at_data5 = np.genfromtxt("DNN_Training/DNN_PX4_Eval5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h5 = at_data5.shape[0]
		at_w5 = at_data5.shape[1]


#		print('original csv size %d x %d' %(all_h, all_w))
		testset_h1 = int((at_h1-length)/step_size)
		testset_h2 = int((at_h2-length)/step_size)
		testset_h3 = int((at_h3-length)/step_size)
		testset_h4 = int((at_h4-length)/step_size)
		testset_h5 = int((at_h5-length)/step_size)

		testset_h = testset_h1+testset_h2+testset_h3+testset_h4+testset_h5


		test_dataset = np.empty((testset_h,length,2))
		index_temp = 0

		for index in range(testset_h1):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data1[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data1[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h1

		for index in range(testset_h2):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data2[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data2[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h2

		for index in range(testset_h3):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data3[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data3[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h3

		for index in range(testset_h4):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data4[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data4[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h4

		for index in range(testset_h5):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data5[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data5[index*step_size+sub_index][9]




#		test_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Eval_attack_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
#		test_h1 = test_data1.shape[0]
#		test_w1 = test_data1.shape[1]
#		test_step_size = 4#10
#		testset_h = int((test_h1-length)/test_step_size)

#		test_data2 = np.genfromtxt("DNN_Training/DNN_PX4_Eval_sensor_combined_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
#		test_h2 = test_data2.shape[0]
#		test_w2 = test_data2.shape[1]

#		test_dataset = np.empty((testset_h,length,2))

#		for index in range(testset_h):
#			for sub_index in range(length):
#				test_dataset[index][sub_index][0] = test_data1[index*test_step_size+sub_index][6]
#				test_dataset[index][sub_index][1] = test_data2[index*test_step_size+sub_index][0]

		np.take(test_dataset, np.random.permutation(test_dataset.shape[0]),axis=0,out=test_dataset)

		test_attack = np.empty((testset_h,length))
		test_gyro = np.empty((testset_h,length))

		for index in range(testset_h):
			for sub_index in range(length):
				test_attack[index][sub_index] = test_dataset[index][sub_index][0]
				test_gyro[index][sub_index] = test_dataset[index][sub_index][1]

	
		test_attack.clip(input_min, input_max)
		test_gyro.clip(data_min, data_max)

		test_attack = (test_attack.astype("float32")-input_median) / input_range + 0.5
		test_gyro = (test_gyro.astype("float32")-data_median) / data_range + 0.5

		if testset_h > int(train_h*0.1):
			testset_h = int(train_h*0.1)
	
		print("Train %d Test %d" %(train_h, testset_h))
		test_input = np.empty((testset_h,length))
		test_output = np.empty((testset_h,length))

		test_input = test_attack[0:testset_h-1,:]
		test_output = test_gyro[0:testset_h-1,:]

		#process valid set
		valid_attack = np.empty((25,length))
		valid_gyro = np.empty((25,length))


		for index in range(25):
			for sub_index in range(length):
				valid_attack[index][sub_index] = test_dataset[index][sub_index][0]
				valid_gyro[index][sub_index] = test_dataset[index][sub_index][1]
		
		valid_attack.clip(input_min, input_max)
		valid_gyro.clip(data_min, data_max)

		valid_attack = (valid_attack.astype("float32")-input_median) / input_range + 0.5
		valid_gyro = (valid_gyro.astype("float32")-data_median) / data_range + 0.5

		return (all_attack, all_gyro, test_input, test_output, valid_attack, valid_gyro)


	def file_load_eval():
		a_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Training1_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h1 = a_data1.shape[0]
		a_w1 = a_data1.shape[1]
		a_data2 = np.genfromtxt("DNN_Training/DNN_PX4_Training2_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h2 = a_data2.shape[0]
		a_w2 = a_data2.shape[1]
		a_data3 = np.genfromtxt("DNN_Training/DNN_PX4_Training3_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h3 = a_data3.shape[0]
		a_w3 = a_data3.shape[1]
		a_data4 = np.genfromtxt("DNN_Training/DNN_PX4_Training4_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h4 = a_data4.shape[0]
		a_w4 = a_data4.shape[1]
		a_data5 = np.genfromtxt("DNN_Training/DNN_PX4_Training5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h5 = a_data5.shape[0]
		a_w5 = a_data5.shape[1]
		a_data6 = np.genfromtxt("DNN_Training/DNN_PX4_Training6_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h6 = a_data6.shape[0]
		a_w6 = a_data6.shape[1]
		a_data7 = np.genfromtxt("DNN_Training/DNN_PX4_Training7_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h7 = a_data7.shape[0]
		a_w7 = a_data7.shape[1]
		a_data8 = np.genfromtxt("DNN_Training/DNN_PX4_Training8_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h8 = a_data8.shape[0]
		a_w8 = a_data8.shape[1]
		a_data9 = np.genfromtxt("DNN_Training/DNN_PX4_Training9_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h9 = a_data9.shape[0]
		a_w9 = a_data9.shape[1]
		a_data10 = np.genfromtxt("DNN_Training/DNN_PX4_Training10_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h10 = a_data10.shape[0]
		a_w10 = a_data10.shape[1]
		a_data11 = np.genfromtxt("DNN_Training/DNN_PX4_Training11_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h11 = a_data11.shape[0]
		a_w11 = a_data11.shape[1]
		a_data12 = np.genfromtxt("DNN_Training/DNN_PX4_Training12_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h12 = a_data12.shape[0]
		a_w12 = a_data12.shape[1]
		a_data13 = np.genfromtxt("DNN_Training/DNN_PX4_Training13_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h13 = a_data13.shape[0]
		a_w13 = a_data13.shape[1]
		a_data14 = np.genfromtxt("DNN_Training/DNN_PX4_Training14_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h14 = a_data14.shape[0]
		a_w14 = a_data14.shape[1]
		a_data15 = np.genfromtxt("DNN_Training/DNN_PX4_Training15_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h15 = a_data15.shape[0]
		a_w15 = a_data15.shape[1]
		a_data16 = np.genfromtxt("DNN_Training/DNN_PX4_Training16_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h16 = a_data16.shape[0]
		a_w16 = a_data16.shape[1]
		a_data17 = np.genfromtxt("DNN_Training/DNN_PX4_Training17_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h17 = a_data17.shape[0]
		a_w17 = a_data17.shape[1]
		a_data18 = np.genfromtxt("DNN_Training/DNN_PX4_Training18_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h18 = a_data18.shape[0]
		a_w18 = a_data18.shape[1]
		a_data19 = np.genfromtxt("DNN_Training/DNN_PX4_Training19_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h19 = a_data19.shape[0]
		a_w19 = a_data19.shape[1]
		a_data20 = np.genfromtxt("DNN_Training/DNN_PX4_Training20_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h20 = a_data20.shape[0]
		a_w20 = a_data20.shape[1]




#		print('original csv size %d x %d' %(all_h, all_w))
		length = 256#1024#128
		step_size = 4#256
		dataset_h1 = int((a_h1-length)/step_size)
		dataset_h2 = int((a_h2-length)/step_size)
		dataset_h3 = int((a_h3-length)/step_size)
		dataset_h4 = int((a_h4-length)/step_size)
		dataset_h5 = int((a_h5-length)/step_size)
		dataset_h6 = int((a_h6-length)/step_size)
		dataset_h7 = int((a_h7-length)/step_size)
		dataset_h8 = int((a_h8-length)/step_size)
		dataset_h9 = int((a_h9-length)/step_size)
		dataset_h10 = int((a_h10-length)/step_size)
		dataset_h11 = int((a_h11-length)/step_size)
		dataset_h12 = int((a_h12-length)/step_size)
		dataset_h13 = int((a_h13-length)/step_size)
		dataset_h14 = int((a_h14-length)/step_size)
		dataset_h15 = int((a_h15-length)/step_size)
		dataset_h16 = int((a_h16-length)/step_size)
		dataset_h17 = int((a_h17-length)/step_size)
		dataset_h18 = int((a_h18-length)/step_size)
		dataset_h19 = int((a_h19-length)/step_size)
		dataset_h20 = int((a_h20-length)/step_size)

		dataset_h = dataset_h1+dataset_h2+dataset_h3+dataset_h4+dataset_h5+dataset_h6+dataset_h7+dataset_h8+dataset_h9+dataset_h10+dataset_h11+dataset_h12+dataset_h13+dataset_h14+dataset_h15+dataset_h16+dataset_h17+dataset_h18+dataset_h19+dataset_h20

		all_dataset = np.empty((dataset_h,length,2))
		index_temp = 0

		for index in range(dataset_h1):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data1[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data1[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h1

		for index in range(dataset_h2):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data2[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data2[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h2

		for index in range(dataset_h3):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data3[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data3[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h3

		for index in range(dataset_h4):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data4[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data4[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h4

		for index in range(dataset_h5):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data5[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data5[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h5

		for index in range(dataset_h6):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data6[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data6[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h6

		for index in range(dataset_h7):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data7[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data7[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h7

		for index in range(dataset_h8):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data8[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data8[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h8

		for index in range(dataset_h9):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data9[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data9[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h9

		for index in range(dataset_h10):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data10[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data10[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h10

		for index in range(dataset_h11):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data11[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data11[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h11

		for index in range(dataset_h12):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data12[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data12[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h12

		for index in range(dataset_h13):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data13[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data13[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h13

		for index in range(dataset_h14):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data14[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data14[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h14

		for index in range(dataset_h15):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data15[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data15[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h15

		for index in range(dataset_h16):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data16[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data16[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h16

		for index in range(dataset_h17):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data17[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data17[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h17

		for index in range(dataset_h18):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data18[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data18[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h18

		for index in range(dataset_h19):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data19[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data19[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h19

		for index in range(dataset_h20):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data20[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data20[index*step_size+sub_index][9]

		np.take(all_dataset, np.random.permutation(all_dataset.shape[0]),axis=0,out=all_dataset)

		all_attack = np.empty((dataset_h,length))
		all_gyro = np.empty((dataset_h,length))

		for index in range(dataset_h):
			for sub_index in range(length):
				all_attack[index][sub_index] = all_dataset[index][sub_index][0]
				all_gyro[index][sub_index] = all_dataset[index][sub_index][1]

		data_max = np.max(all_gyro)
		data_min = np.min(all_gyro)
		data_range = (data_max - data_min)
		data_median = (data_max + data_min)/2

		input_max = np.max(all_attack)
		input_min = np.min(all_attack)
		input_range = (input_max - input_min)
		input_median = (input_max + input_min)/2

		upper_bound = input_max - data_range
		lower_bound = input_min + data_range

		##test set processing
		at_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Eval1_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h1 = at_data1.shape[0]
		at_w1 = at_data1.shape[1]
		at_data2 = np.genfromtxt("DNN_Training/DNN_PX4_Eval2_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h2 = at_data2.shape[0]
		at_w2 = at_data2.shape[1]
		at_data3 = np.genfromtxt("DNN_Training/DNN_PX4_Eval3_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h3 = at_data3.shape[0]
		at_w3 = at_data3.shape[1]
		at_data4 = np.genfromtxt("DNN_Training/DNN_PX4_Eval4_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h4 = at_data4.shape[0]
		at_w4 = at_data4.shape[1]
		at_data5 = np.genfromtxt("DNN_Training/DNN_PX4_Eval5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h5 = at_data5.shape[0]
		at_w5 = at_data5.shape[1]


#		print('original csv size %d x %d' %(all_h, all_w))
		testset_h1 = int((at_h1-length)/step_size)
		testset_h2 = int((at_h2-length)/step_size)
		testset_h3 = int((at_h3-length)/step_size)
		testset_h4 = int((at_h4-length)/step_size)
		testset_h5 = int((at_h5-length)/step_size)

		testset_h = testset_h1+testset_h2+testset_h3+testset_h4+testset_h5


		test_dataset = np.empty((testset_h,length,2))
		index_temp = 0

		for index in range(testset_h1):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data1[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data1[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h1

		for index in range(testset_h2):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data2[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data2[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h2

		for index in range(testset_h3):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data3[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data3[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h3

		for index in range(testset_h4):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data4[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data4[index*step_size+sub_index][9]
		index_temp = index_temp+testset_h4

		for index in range(testset_h5):
			for sub_index in range(length):
				test_dataset[index+index_temp][sub_index][0] = at_data5[index*step_size+sub_index][6]
				test_dataset[index+index_temp][sub_index][1] = at_data5[index*step_size+sub_index][9]



		np.take(test_dataset, np.random.permutation(test_dataset.shape[0]),axis=0,out=test_dataset)


		#process valid set
		valid_attack = np.empty((25,length))
		valid_gyro = np.empty((25,length))


		for index in range(25):
			for sub_index in range(length):
				valid_attack[index][sub_index] = test_dataset[index][sub_index][0]
				valid_gyro[index][sub_index] = test_dataset[index][sub_index][1]
		
		valid_attack.clip(input_min, input_max)
		valid_gyro.clip(data_min, data_max)

		valid_attack = (valid_attack.astype("float32")-input_median) / input_range + 0.5
		valid_gyro = (valid_gyro.astype("float32")-data_median) / data_range + 0.5

		return (valid_attack, valid_gyro)

	def file_load_test():
		a_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Training1_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h1 = a_data1.shape[0]
		a_w1 = a_data1.shape[1]
		a_data2 = np.genfromtxt("DNN_Training/DNN_PX4_Training2_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h2 = a_data2.shape[0]
		a_w2 = a_data2.shape[1]
		a_data3 = np.genfromtxt("DNN_Training/DNN_PX4_Training3_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h3 = a_data3.shape[0]
		a_w3 = a_data3.shape[1]
		a_data4 = np.genfromtxt("DNN_Training/DNN_PX4_Training4_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h4 = a_data4.shape[0]
		a_w4 = a_data4.shape[1]
		a_data5 = np.genfromtxt("DNN_Training/DNN_PX4_Training5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h5 = a_data5.shape[0]
		a_w5 = a_data5.shape[1]
		a_data6 = np.genfromtxt("DNN_Training/DNN_PX4_Training6_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h6 = a_data6.shape[0]
		a_w6 = a_data6.shape[1]
		a_data7 = np.genfromtxt("DNN_Training/DNN_PX4_Training7_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h7 = a_data7.shape[0]
		a_w7 = a_data7.shape[1]
		a_data8 = np.genfromtxt("DNN_Training/DNN_PX4_Training8_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h8 = a_data8.shape[0]
		a_w8 = a_data8.shape[1]
		a_data9 = np.genfromtxt("DNN_Training/DNN_PX4_Training9_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h9 = a_data9.shape[0]
		a_w9 = a_data9.shape[1]
		a_data10 = np.genfromtxt("DNN_Training/DNN_PX4_Training10_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h10 = a_data10.shape[0]
		a_w10 = a_data10.shape[1]
		a_data11 = np.genfromtxt("DNN_Training/DNN_PX4_Training11_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h11 = a_data11.shape[0]
		a_w11 = a_data11.shape[1]
		a_data12 = np.genfromtxt("DNN_Training/DNN_PX4_Training12_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h12 = a_data12.shape[0]
		a_w12 = a_data12.shape[1]
		a_data13 = np.genfromtxt("DNN_Training/DNN_PX4_Training13_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h13 = a_data13.shape[0]
		a_w13 = a_data13.shape[1]
		a_data14 = np.genfromtxt("DNN_Training/DNN_PX4_Training14_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h14 = a_data14.shape[0]
		a_w14 = a_data14.shape[1]
		a_data15 = np.genfromtxt("DNN_Training/DNN_PX4_Training15_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h15 = a_data15.shape[0]
		a_w15 = a_data15.shape[1]
		a_data16 = np.genfromtxt("DNN_Training/DNN_PX4_Training16_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h16 = a_data16.shape[0]
		a_w16 = a_data16.shape[1]
		a_data17 = np.genfromtxt("DNN_Training/DNN_PX4_Training17_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h17 = a_data17.shape[0]
		a_w17 = a_data17.shape[1]
		a_data18 = np.genfromtxt("DNN_Training/DNN_PX4_Training18_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h18 = a_data18.shape[0]
		a_w18 = a_data18.shape[1]
		a_data19 = np.genfromtxt("DNN_Training/DNN_PX4_Training19_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h19 = a_data19.shape[0]
		a_w19 = a_data19.shape[1]
		a_data20 = np.genfromtxt("DNN_Training/DNN_PX4_Training20_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h20 = a_data20.shape[0]
		a_w20 = a_data20.shape[1]




#		print('original csv size %d x %d' %(all_h, all_w))
		length = 256#1024#128
		step_size = 4#256
		dataset_h1 = int((a_h1-length)/step_size)
		dataset_h2 = int((a_h2-length)/step_size)
		dataset_h3 = int((a_h3-length)/step_size)
		dataset_h4 = int((a_h4-length)/step_size)
		dataset_h5 = int((a_h5-length)/step_size)
		dataset_h6 = int((a_h6-length)/step_size)
		dataset_h7 = int((a_h7-length)/step_size)
		dataset_h8 = int((a_h8-length)/step_size)
		dataset_h9 = int((a_h9-length)/step_size)
		dataset_h10 = int((a_h10-length)/step_size)
		dataset_h11 = int((a_h11-length)/step_size)
		dataset_h12 = int((a_h12-length)/step_size)
		dataset_h13 = int((a_h13-length)/step_size)
		dataset_h14 = int((a_h14-length)/step_size)
		dataset_h15 = int((a_h15-length)/step_size)
		dataset_h16 = int((a_h16-length)/step_size)
		dataset_h17 = int((a_h17-length)/step_size)
		dataset_h18 = int((a_h18-length)/step_size)
		dataset_h19 = int((a_h19-length)/step_size)
		dataset_h20 = int((a_h20-length)/step_size)

		dataset_h = dataset_h1+dataset_h2+dataset_h3+dataset_h4+dataset_h5+dataset_h6+dataset_h7+dataset_h8+dataset_h9+dataset_h10+dataset_h11+dataset_h12+dataset_h13+dataset_h14+dataset_h15+dataset_h16+dataset_h17+dataset_h18+dataset_h19+dataset_h20

		all_dataset = np.empty((dataset_h,length,2))
		index_temp = 0

		for index in range(dataset_h1):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data1[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data1[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h1

		for index in range(dataset_h2):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data2[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data2[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h2

		for index in range(dataset_h3):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data3[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data3[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h3

		for index in range(dataset_h4):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data4[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data4[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h4

		for index in range(dataset_h5):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data5[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data5[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h5

		for index in range(dataset_h6):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data6[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data6[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h6

		for index in range(dataset_h7):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data7[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data7[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h7

		for index in range(dataset_h8):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data8[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data8[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h8

		for index in range(dataset_h9):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data9[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data9[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h9

		for index in range(dataset_h10):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data10[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data10[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h10

		for index in range(dataset_h11):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data11[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data11[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h11

		for index in range(dataset_h12):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data12[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data12[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h12

		for index in range(dataset_h13):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data13[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data13[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h13

		for index in range(dataset_h14):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data14[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data14[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h14

		for index in range(dataset_h15):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data15[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data15[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h15

		for index in range(dataset_h16):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data16[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data16[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h16

		for index in range(dataset_h17):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data17[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data17[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h17

		for index in range(dataset_h18):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data18[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data18[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h18

		for index in range(dataset_h19):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data19[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data19[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h19

		for index in range(dataset_h20):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data20[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data20[index*step_size+sub_index][9]

		np.take(all_dataset, np.random.permutation(all_dataset.shape[0]),axis=0,out=all_dataset)

		all_attack = np.empty((dataset_h,length))
		all_gyro = np.empty((dataset_h,length))

		for index in range(dataset_h):
			for sub_index in range(length):
				all_attack[index][sub_index] = all_dataset[index][sub_index][0]
				all_gyro[index][sub_index] = all_dataset[index][sub_index][1]

		data_max = np.max(all_gyro)
		data_min = np.min(all_gyro)
		data_range = (data_max - data_min)
		data_median = (data_max + data_min)/2

		input_max = np.max(all_attack)
		input_min = np.min(all_attack)
		input_range = (input_max - input_min)
		input_median = (input_max + input_min)/2

		upper_bound = input_max - data_range
		lower_bound = input_min + data_range

		##test set processing

		at_data1 = np.genfromtxt("DNN_Training/DNN_PX4_Test5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		at_h1 = at_data1.shape[0]
		at_w1 = at_data1.shape[1]

		testset_h1 = int((at_h1-length))#/step_size)
		testset_h = testset_h1#+testset_h2+testset_h3+testset_h4+testset_h5

		test_attack = np.empty((testset_h,length))
		test_gyro = np.empty((testset_h,length))

		for index in range(testset_h1):
			for sub_index in range(length):
				test_attack[index][sub_index] = at_data1[index+sub_index][6]
				test_gyro[index][sub_index] = at_data1[index+sub_index][9]
		
		test_attack.clip(input_min, input_max)
		test_gyro.clip(data_min, data_max)

		test_attack = (test_attack.astype("float32")-input_median) / input_range + 0.5
		test_gyro = (test_gyro.astype("float32")-data_median) / data_range + 0.5

		return (test_attack, test_gyro, input_min, input_max, input_median, data_min, data_max, data_median, testset_h )




	def file_data_range():
		a_data1 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training1_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h1 = a_data1.shape[0]
		a_w1 = a_data1.shape[1]
		a_data2 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training2_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h2 = a_data2.shape[0]
		a_w2 = a_data2.shape[1]
		a_data3 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training3_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h3 = a_data3.shape[0]
		a_w3 = a_data3.shape[1]
		a_data4 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training4_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h4 = a_data4.shape[0]
		a_w4 = a_data4.shape[1]
		a_data5 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training5_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h5 = a_data5.shape[0]
		a_w5 = a_data5.shape[1]
		a_data6 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training6_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h6 = a_data6.shape[0]
		a_w6 = a_data6.shape[1]
		a_data7 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training7_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h7 = a_data7.shape[0]
		a_w7 = a_data7.shape[1]
		a_data8 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training8_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h8 = a_data8.shape[0]
		a_w8 = a_data8.shape[1]
		a_data9 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training9_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h9 = a_data9.shape[0]
		a_w9 = a_data9.shape[1]
		a_data10 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training10_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h10 = a_data10.shape[0]
		a_w10 = a_data10.shape[1]
		a_data11 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training11_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h11 = a_data11.shape[0]
		a_w11 = a_data11.shape[1]
		a_data12 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training12_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h12 = a_data12.shape[0]
		a_w12 = a_data12.shape[1]
		a_data13 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training13_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h13 = a_data13.shape[0]
		a_w13 = a_data13.shape[1]
		a_data14 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training14_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h14 = a_data14.shape[0]
		a_w14 = a_data14.shape[1]
		a_data15 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training15_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h15 = a_data15.shape[0]
		a_w15 = a_data15.shape[1]
		a_data15 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training15_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h15 = a_data15.shape[0]
		a_w15 = a_data15.shape[1]
		a_data16 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training16_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h16 = a_data16.shape[0]
		a_w16 = a_data16.shape[1]
		a_data17 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training17_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h17 = a_data17.shape[0]
		a_w17 = a_data17.shape[1]
		a_data18 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training18_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h18 = a_data18.shape[0]
		a_w18 = a_data18.shape[1]
		a_data19 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training19_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h19 = a_data19.shape[0]
		a_w19 = a_data19.shape[1]
		a_data20 = np.genfromtxt("/home/cyber040946/SITL_Recovery/accel_autoencoder_1D_PX4_SITL_MA/DNN_Training/DNN_PX4_Training20_attacc_status_0.csv",delimiter=",",dtype=np.float32, skip_header=True)[:,1:]
		a_h20 = a_data20.shape[0]
		a_w20 = a_data20.shape[1]




#		print('original csv size %d x %d' %(all_h, all_w))
		length = 256#1024#128
		step_size = 256#10
		dataset_h1 = int((a_h1-length)/step_size)
		dataset_h2 = int((a_h2-length)/step_size)
		dataset_h3 = int((a_h3-length)/step_size)
		dataset_h4 = int((a_h4-length)/step_size)
		dataset_h5 = int((a_h5-length)/step_size)
		dataset_h6 = int((a_h6-length)/step_size)
		dataset_h7 = int((a_h7-length)/step_size)
		dataset_h8 = int((a_h8-length)/step_size)
		dataset_h9 = int((a_h9-length)/step_size)
		dataset_h10 = int((a_h10-length)/step_size)
		dataset_h11 = int((a_h11-length)/step_size)
		dataset_h12 = int((a_h12-length)/step_size)
		dataset_h13 = int((a_h13-length)/step_size)
		dataset_h14 = int((a_h14-length)/step_size)
		dataset_h15 = int((a_h15-length)/step_size)
		dataset_h16 = int((a_h16-length)/step_size)
		dataset_h17 = int((a_h17-length)/step_size)
		dataset_h18 = int((a_h18-length)/step_size)
		dataset_h19 = int((a_h19-length)/step_size)
		dataset_h20 = int((a_h20-length)/step_size)

		dataset_h = dataset_h1+dataset_h2+dataset_h3+dataset_h4+dataset_h5+dataset_h6+dataset_h7+dataset_h8+dataset_h9+dataset_h10+dataset_h11+dataset_h12+dataset_h13+dataset_h14+dataset_h15+dataset_h16+dataset_h17+dataset_h18+dataset_h19+dataset_h20


		all_dataset = np.empty((dataset_h,length,2))
		index_temp = 0
                
                
		for index in range(dataset_h1):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data1[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data1[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h1

		for index in range(dataset_h2):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data2[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data2[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h2

		for index in range(dataset_h3):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data3[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data3[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h3

		for index in range(dataset_h4):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data4[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data4[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h4

		for index in range(dataset_h5):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data5[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data5[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h5

		for index in range(dataset_h6):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data6[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data6[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h6

		for index in range(dataset_h7):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data7[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data7[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h7

		for index in range(dataset_h8):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data8[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data8[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h8

		for index in range(dataset_h9):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data9[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data9[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h9

		for index in range(dataset_h10):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data10[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data10[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h10

		for index in range(dataset_h11):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data11[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data11[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h11

		for index in range(dataset_h12):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data12[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data12[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h12

		for index in range(dataset_h13):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data13[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data13[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h13

		for index in range(dataset_h14):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data14[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data14[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h14

		for index in range(dataset_h15):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data15[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data15[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h15

		for index in range(dataset_h16):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data16[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data16[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h16

		for index in range(dataset_h17):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data17[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data17[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h17

		for index in range(dataset_h18):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data18[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data18[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h18

		for index in range(dataset_h19):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data19[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data19[index*step_size+sub_index][9]
		index_temp = index_temp+dataset_h19

		for index in range(dataset_h20):
			for sub_index in range(length):
				all_dataset[index+index_temp][sub_index][0] = a_data20[index*step_size+sub_index][6]
				all_dataset[index+index_temp][sub_index][1] = a_data20[index*step_size+sub_index][9]


		np.take(all_dataset, np.random.permutation(all_dataset.shape[0]),axis=0,out=all_dataset)

		all_attack = np.empty((dataset_h,length))
		all_gyro = np.empty((dataset_h,length))

		for index in range(dataset_h):
			for sub_index in range(length):
				all_attack[index][sub_index] = all_dataset[index][sub_index][0]
				all_gyro[index][sub_index] = all_dataset[index][sub_index][1]

		data_max = np.max(all_gyro)
		data_min = np.min(all_gyro)
		data_range = (data_max - data_min)
		data_median = (data_max + data_min)/2

		input_max = np.max(all_attack)
		input_min = np.min(all_attack)
		input_range = (input_max - input_min)
		input_median = (input_max + input_min)/2

		upper_bound = input_max - data_range
		lower_bound = input_min + data_range

		print('Raw Data(min,max,median):%f %f %f \n True Data(min,max,median):%f %f %f' 
                        %(input_min, input_max, input_median, data_min, data_max, data_median))
		return (input_min, input_max, input_median, data_min, data_max, data_median)





