#!/usr/bin/env python

# USAGE
# python train_conv_autoencoder.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import socket
import errno
import sys
import signal
import struct
import pexpect
import time
import select
import threading
import ctypes
import multiprocessing
import os
#from threading import Thread
from multiprocessing import Process, Queue, Value, Array
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink2
from pymavlink.dialects.v20 import custom_messages as mavlink3

EPOCHS = 25
BS = 32
LENGTH = 256#512#1024#128
DIALECT = 'custom'
time_log = 0
global return_value, pr_com


def interpret_address(addrstr):
    """Interpret a IP:port string."""
    a = addrstr.split(':')
    a[1] = int(a[1])
    return tuple(a)

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())


def process_com(v_start, v_end, v_copy, v_trigger, v_inf_first, a_input, a_output, v_re_tr):
    
    inf_in = mavutil.mavlink_connection('udpin:localhost:14548', dialect = DIALECT)
    inf_in.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (inf_in.target_system, inf_in.target_component))
    recv_count = 0
    time_log = 0
    r_value = 0
    
    input_dataset = np.empty(LENGTH)
    result_data = np.empty(LENGTH)

    print('p_com: process com started')
    inf_in.mav.request_data_stream_send(inf_in.target_system, inf_in.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 400, 1)


    while True:
        start_time = time.process_time()
        msg = inf_in.recv_match(type='DNN_SEND',blocking=True, timeout=1)

        if not msg:
            print("Nothing")
            v_re_tr.value = -1
            return
        if msg.get_type() == "BAD_DATA":
            print("Bad data")
            if mavutil.all_printable(msg.data):
                sys.stdout.write(msg.data)
                sys.stdout.flush()
        else:
            v_trigger.value = msg.attack_trigger
            raw_signal = msg.compromised_signal

            if(v_trigger.value == 1):
                if recv_count < LENGTH:
                    input_dataset[recv_count] = raw_signal
                    recv_count = recv_count + 1
                    if recv_count == LENGTH:
                        if (v_start.value == 0) and (v_end.value == 0):
                            a_input[:] = input_dataset[:]
                            v_start.value = 1
                    recovered = 0
                else:
                    v_start.value = 1
                    input_dataset[0:LENGTH-1] = input_dataset[1:LENGTH] 
                    input_dataset[LENGTH-1] = raw_signal
                    recv_count = LENGTH *2
                    if v_inf_first.value == 0:
                        recovered = 0
                    else:
                        if v_end.value == 1:
                            temp_count2 = 0
                            result_data = np.frombuffer(a_output.get_obj())
                            a_input[:] = input_dataset[:]
                            v_end.value = 0
                            time_log = 1
                        recovered = result_data[LENGTH+temp_count2-7]
                        temp_count2 = temp_count2 + 1
                        temp_count2 = temp_count2%7#limit max index
            else:
                recovered = raw_signal
                v_start.value = 0

            inf_in.mav.dnn_recv_send(recovered)
        end_time =time.process_time()

        if(time_log == 1):
            time_log = 0

def process_com_inf(v_start, v_end, v_copy, v_trigger, v_inf_first, a_input, a_output):
    com_out_address = interpret_address("127.0.0.1:9011")
    com_in_address = interpret_address("127.0.0.1:9012")
    com_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    com_in.bind(com_in_address)
    com_in.setblocking(0) 
    com_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    com_out.connect(com_out_address)
    com_out.setblocking(0)

    com_out_buf = bytes()

    print('p_inf: process com inf started')

    while True:

        if (v_trigger.value == 1) and (v_start.value == 1) and (v_end.value == 0):
            start_time = time.process_time()
            com_out_buf = struct.pack('f', a_input[0])

            for i in range(LENGTH-1):
                com_out_buf += struct.pack('f', a_input[i+1])

            try:
                com_out.send(com_out_buf)
            except socket.error as e:
                if e.errno not in [errno.ECONNREFUSED]:
                    raise
                
            rin = [com_in.fileno()]
            try:
                (rin, win, xin) = select.select(rin, [], [], 1.0)
            
            except select.error:
                util.check_parent()
                continue
            if com_in.fileno() in rin:
                com_in_buf = com_in.recv(LENGTH*4)
                temp = struct.unpack('f'*LENGTH, com_in_buf)
                for i in range(LENGTH):
                    a_output[i] = temp[i]
                v_end.value = 1
                end_time =time.process_time()
                v_inf_first.value = 1##

def process_restart(v_re_tr, pr_com):
    print("Restart Process Run")
    while True:
        time.sleep(2)
        print("%d" % v_re_tr.value)
        if v_re_tr.value == -1:
            print("Restart Pr com")

            pr_com.terminate()
            time.sleep(3)
            pr_com.start()
            pr_com.join()

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type=int, default=8,
	help="# number of samples to visualize when decoding")
ap.add_argument("-o", "--output", type=str, default="output.png",
	help="path to output visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output plot file")
args = vars(ap.parse_args())

# initialize the number of epochs to train for and batch size

print("[INFO] making predictions...")



def main_loop():
    """Run main loop."""
   
    v_start = Value('i', 0)
    v_end = Value('i', 0)
    v_copy = Value('i', 0)

    v_trigger = Value('i', 0)
    v_inf_first = Value('i', 0)
    
    v_re_tr = Value('i',0)
    
    a_input = Array("d",LENGTH)
    a_output = Array("d",LENGTH)

    pr_list = []
    return_value = 0
    print('Start Main loop')

    pr_com = Process(target=process_com, args=(v_start,v_end,v_copy,v_trigger,v_inf_first,a_input,a_output,v_re_tr))
    pr_inf = Process(target=process_com_inf, args=(v_start,v_end,v_copy,v_trigger,v_inf_first,a_input,a_output))

    pr_com.daemon = True
    pr_inf.daemon = True

    pr_list.append(pr_com)
    pr_list.append(pr_inf)

    pr_com.start()
    pr_inf.start()
    pr_com.join()
    pr_inf.join()


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


