#!/usr/bin/env python

# USAGE
# python train_conv_autoencoder.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.convautoencoder_1D import ConvAutoencoder_1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.compat.v1.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import socket
import errno
import sys
#import fdpexpect
import signal
import struct
import pexpect
import time
import select

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow.contrib import tensorrt as trt
# TF-TRT related library
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2
from pymavlink.dialects.v20 import custom_messages as mavlink3



DIALECT = 'custom'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction =0.8

session = tf.compat.v1.Session(config=config)
K.set_session(session)

EPOCHS = 25
BS = 32
LENGTH = 256

recv_count = 0
input_dataset = np.empty((1,LENGTH,1))
input_data = np.empty((1,LENGTH,1))
out_file = open('inference_time.txt', 'w')

saved_model_loaded = tf.saved_model.load('/home/cyber040946/SITL_Recovery/gyro_autoencoder_1D_PX4_SITL_MA/model2', tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_variables_to_constants_v2(graph_func)


def interpret_address(addrstr):
    """Interpret a IP:port string."""
    a = addrstr.split(':')
    a[1] = int(a[1])
    return tuple(a)

def run_inference(msg):
    """Process FG FDM input from JSBSim."""
    global input_min, input_max, input_median, data_min, data_max, data_median, input_range, data_range, recv_count, input_dataset, out_data, frozen_func
    global input_data
    trigger = msg.attack_trigger
    raw_signal = msg.compromised_signal
    send_timestamp = msg.time_usec
    if(trigger == 1):
        if recv_count < LENGTH:
            input_dataset[0][recv_count][0] = raw_signal
            recv_count = recv_count + 1
            if recv_count == LENGTH:
                input_data = (input_dataset.clip(input_min, input_max)-input_median) / input_range + 0.5
                out_data = frozen_func(tf.convert_to_tensor(input_data.astype("float32")))[0].numpy()
            recovered = 0.0
        else:
            for i in range(LENGTH-1):
                input_dataset[0][i][0] = input_dataset[0][i+1][0]
            input_dataset[0][LENGTH-1][0] = raw_signal
            recv_count = 100000
            input_data = (input_dataset.clip(input_min, input_max)-input_median) / input_range + 0.5
            start_time_infer = time.process_time()
            out_data = frozen_func(tf.convert_to_tensor(input_data.astype("float32")))[0].numpy()
            end_time_infer =time.process_time()
            out_file.write('inference time: %f\n' %(end_time_infer-start_time_infer))
            out_data = (out_data-0.5)*data_range + data_median
            recovered = out_data[0][LENGTH-1][0]
    else:
        recovered = raw_signal
        recv_count = 0
    inf_in.mav.dnn_recv_send(recovered, send_timestamp)
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8,
	help="# number of samples to visualize when decoding")
ap.add_argument("-o", "--output", type=str, default="output.png",
	help="path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")
args = vars(ap.parse_args())

# initialize the number of epochs to train for and batch size

(input_min, input_max, input_median, data_min, data_max, data_median) = ConvAutoencoder_1D.file_data_range()
data_range = (data_max - data_min)
input_range = (input_max - input_min)

print("[INFO] making predictions...")


inf_in = mavutil.mavlink_connection('udpin:localhost:14548', dialect = DIALECT)
inf_in.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (inf_in.target_system, inf_in.target_component))
inf_in.mav.request_data_stream_send(inf_in.target_system, inf_in.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 400, 1)




def main_loop():
    """Run main loop."""
    tnow = time.time()
    last_report = tnow
    last_sim_input = tnow
    last_wind_update = tnow
    connected = 0

    print('Start Main loop')
    while True:
        msg = inf_in.recv_match(type='DNN_SEND',blocking=True, timeout=1)
        if not msg:
            print("Nothing")
            return
        if msg.get_type() == "BAD_DATA":
            print("Bad data")
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()
        else:
            run_inference(msg)

def exit_handler():
    """Exit the sim."""
    print("running exit handler")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    out_file.close()
    sys.exit(1)

signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGTERM, exit_handler)

try:
    main_loop()
except Exception as ex:
    print(ex)
    exit_handler()
    raise


