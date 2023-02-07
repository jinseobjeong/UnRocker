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
import threading
import ctypes
import threading

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import InteractiveSession
# TF-TRT related library
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from threading import Thread
from multiprocessing import Process, Queue, Value, Array

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction =0.8

session = tf.compat.v1.InteractiveSession(config=config)

EPOCHS = 25
BS = 32
LENGTH = 256

inf_start = 0
inf_end = 0
inf_first = 0
packet_count = 0
trigger = 0
raw_signal = 0.0
packet_recv = 0
input_dataset = np.empty((1,LENGTH,1))
input_data = np.empty((1,LENGTH,1))
out_data = np.empty((1,LENGTH,1))
com_out_buf = bytes()


saved_model_loaded = tf.saved_model.load('./model2', tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_variables_to_constants_v2(graph_func)


def interpret_address(addrstr):
    """Interpret a IP:port string."""
    a = addrstr.split(':')
    a[1] = int(a[1])
    return tuple(a)

def run_inference(com_in_buf):
    """Process FG FDM input from JSBSim."""
    global inf_out, input_min, input_max, input_median, data_min, data_max, data_median, input_range, data_range, input_dataset, out_data, frozen_func, com_out_buf
    global input_data
    start_time = time.process_time()
    temp = struct.unpack('f'*LENGTH, com_in_buf)
    for i in range(LENGTH):
        input_dataset[0][i][0] = temp[i]
    input_data = (input_dataset.clip(input_min, input_max)-data_median) / input_range + 0.5
    out_data = frozen_func(tf.convert_to_tensor(input_data.astype("float32")))[0].numpy()
    out_data = (out_data-0.5)*data_range + data_median
    com_out_buf = struct.pack('f', out_data[0][0][0])

    for i in range(LENGTH-1):
        com_out_buf += struct.pack('f', out_data[0][i+1][0])
    end_time =time.process_time()

    try:
        inf_out.send(com_out_buf)
    except socket.error as e:
        if e.errno not in [errno.ECONNREFUSED]:
            raise

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

(input_min, input_max, input_median, data_min, data_max, data_median) = ConvAutoencoder_1D.file_data_range()
data_range = (data_max - data_min)
input_range = (input_max - input_min)


print("[INFO] making predictions...")


inf_out_address = interpret_address("127.0.0.1:9012")
inf_in_address = interpret_address("127.0.0.1:9011")
inf_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
inf_in.bind(inf_in_address)
inf_in.setblocking(0)
inf_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
inf_out.connect(inf_out_address)
inf_out.setblocking(0)


def main_loop():
    """Run main loop."""
    tnow = time.time()
    last_report = tnow
    last_sim_input = tnow
    last_wind_update = tnow
    connected = 0
   
    print('Start Inference Main loop')
    
    while True:
        start_time = time.process_time()
        rin = [inf_in.fileno()]
        try:
            (rin, win, xin) = select.select(rin, [], [], 1.0)
        except select.error:
            util.check_parent()
            continue

        if inf_in.fileno() in rin:
            inf_buf = inf_in.recv(LENGTH*4)
            run_inference(inf_buf)
        end_time =time.process_time()


def exit_handler():
    """Exit the sim."""
    print("running exit handler")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.exit(1)

signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGTERM, exit_handler)

try:
    main_loop()
except Exception as ex:
    print(ex)
    exit_handler()
    raise


