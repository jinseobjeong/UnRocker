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

session = tf.compat.v1.Session(config=config)#InteractiveSession(config=config)
K.set_session(session)

EPOCHS = 25
BS = 32
LENGTH = 256#1024#512#1024#128

recv_count = 0
input_dataset = np.empty((1,LENGTH,1))
input_data = np.empty((1,LENGTH,1))##??necessary??
out_file = open('/home/cyber040946/SITL_Recovery/gyro_autoencoder_1D_PX4_SITL_MA/inference_time.txt', 'w')

#saved_model_loaded = tf.saved_model.load('/home/cyber040946/SITL_Recovery/gyro_autoencoder_1D_PX4_SITL_MA/model2', tags=[tag_constants.SERVING])
#saved_model_loaded = tf.saved_model.load('/home/sysseclab/prj_jsjeong/gyro_autoencoder_1D_PX4_SITL_MA/model2_256_SOLO', tags=[tag_constants.SERVING])
#saved_model_loaded = tf.saved_model.load('/home/sysseclab/prj_jsjeong/gyro_autoencoder_1D_PX4_SITL_MA/model2_128_v2', tags=[tag_constants.SERVING])
#saved_model_loaded = tf.saved_model.load('/home/sysseclab/prj_jsjeong/gyro_autoencoder_1D_PX4_SITL_MA/model2_256', tags=[tag_constants.SERVING])

#saved_model_loaded = tf.saved_model.load('/home/sysseclab/prj_jsjeong/gyro_autoencoder_1D_PX4_SITL_MA/model2_256_IRIS_nodelay', tags=[tag_constants.SERVING])
#graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#frozen_func = convert_variables_to_constants_v2(graph_func)

(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D.build(LENGTH)#10x10 insteadof 28x28

eval_model.load_weights('/home/cyber040946/SITL_Recovery/gyro_autoencoder_1D_PX4_SITL_MA/checkpoints/my_checkpoint')

def interpret_address(addrstr):
    """Interpret a IP:port string."""
    a = addrstr.split(':')
    a[1] = int(a[1])
    return tuple(a)

def run_inference(msg):
    """Process FG FDM input from JSBSim."""
    global input_min, input_max, input_median, data_min, data_max, data_median, input_range, data_range, recv_count, input_dataset, out_data, eval_model#frozen_func
    global input_data
    trigger = msg.attack_trigger#packet[0]
    raw_signal = msg.compromised_signal# packet[1]
    if(trigger == 1):#initialize the input array
        if recv_count < LENGTH:
            input_dataset[0][recv_count][0] = raw_signal
            recv_count = recv_count + 1
            if recv_count == LENGTH:
                input_data = (input_dataset.clip(input_min, input_max)-input_median) / input_range + 0.5
                out_data = eval_model.predict(input_data)#frozen_func(tf.convert_to_tensor(input_data.astype("float32")))[0].numpy()
            recovered = raw_signal

        else:
            for i in range(LENGTH-1):
                input_dataset[0][i][0] = input_dataset[0][i+1][0]
            input_dataset[0][LENGTH-1][0] = raw_signal
            recv_count = 100000#LENGTH*2/
            input_data = (input_dataset.clip(input_min, input_max)-input_median) / input_range + 0.5
            start_time_infer = time.process_time()
            out_data = eval_model.predict(input_data)#frozen_func(tf.convert_to_tensor(input_data.astype("float32")))[0].numpy()
            end_time_infer =time.process_time()
            #print('inference time:', (end_time_infer-start_time_infer))
            out_file.write('inference time: %f\n' %(end_time_infer-start_time_infer))
            #out_data = (out_data-0.5)*data_range + data_median#+0.03#compensation
            recovered = (out_data[0][LENGTH-1][0] -0.5)*data_range + data_median
        #print('attack_triggered')
        #print('gyro value %f' % raw_signal)
    else:
        recovered = raw_signal
        recv_count = 0
    inf_in.mav.dnn_recv_send(recovered)
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

#(valX, valY) = ConvAutoencoder_1D.file_load_eval()

#valX = np.expand_dims(valX, axis=-1)
#valY = np.expand_dims(valY, axis=-1)


print("[INFO] making predictions...")

##
#(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D_light.build(LENGTH)#10x10 insteadof 28x28
#(eval_encoder, eval_decoder, eval_model) = ConvAutoencoder_1D.build(LENGTH)#10x10 insteadof 28x28
#eval_model.load_weights('./checkpoints_200421/my_checkpoint')
#eval_model.load_weights('./checkpoints_200412/my_checkpoint')
#eval_model.load_weights('./checkpoints_200512/my_checkpoint')
#decoded2 = eval_model.predict(valX)


#jsjeong inference socket
#inf_out_address = interpret_address("127.0.0.1:9009")
#inf_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#inf_out.connect(inf_out_address)
#inf_out.setblocking(0)

inf_in = mavutil.mavlink_connection('udpin:localhost:14548', dialect = DIALECT)
inf_in.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (inf_in.target_system, inf_in.target_component))
inf_in.mav.request_data_stream_send(inf_in.target_system, inf_in.target_component, mavutil.mavlink.MAV_DATA_STREAM_ALL, 400, 1)



#for i in range(0, 25):
#	N = np.arange(0, LENGTH)
#	index_n = i
#	#subplot_id = 661+i
#	plt.figure()#subplot(5,5,i+1)
#	plt.style.use("ggplot")
#	plt.plot(N, valX[index_n], label="input_data")
#	plt.plot(N, valY[index_n], label="true_data")
#	plt.plot(N, decoded2[index_n], label="decoded_data")
#	plt.title("Original & recovered data")
#	plt.xlabel("Times")
#	plt.ylabel("Value")
#	plt.ylim((-1,2))
#	plt.legend(loc="lower left")
#	graph = "./eval_results/graph_eval%d" %i
#	plt.savefig(graph)#args["graph"])# %i])



#plt.figure()
#for i in range(0, 25):
#	N = np.arange(0, LENGTH)
#	index_n = i
#	plt.subplot(5,5,i+1)
##	plt.style.use("ggplot")
#	plt.plot(N, valX[index_n], label="input_data")
#	plt.plot(N, valY[index_n], label="true_data")
#	plt.plot(N, decoded2[index_n], label="decoded_data")
#	plt.ylim((-1,2))
#graph = "./eval_results/graph_sum_eval"
#plt.savefig(graph)#args["graph"])# %i])


def main_loop():
    """Run main loop."""
    tnow = time.time()
    last_report = tnow
    last_sim_input = tnow
    last_wind_update = tnow
    connected = 0

#    config = tf.compat.v1.ConfigProto()
#    config.gpu_options.allow_growth = True
#    #config.gpu_options.per_process_gpu_memory_fraction =0.8
#    session = InteractiveSession(config=config)

#    tf.compat.v1.keras.backend.set_session(session)#tf.compat.v1.Session(config=config))
#session = InteractiveSession(config=config)


#    frame_count = 0
#    paused = False
#    simstep = 1.0/opts.rate
#    simtime = simstep
#    frame_time = 1.0/opts.rate
#    scaled_frame_time = frame_time/opts.speedup
#    last_wall_time = time.time()
#    achieved_rate = opts.speedup
    print('Start Main loop')
    while True:
     #   start_time = time.process_time()
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
#            process_sitl_input(simbuf)
    #    end_time =time.process_time()
    #    print('working time:', (end_time-start_time))
#        rin = [jsb_in.fileno(), sim_in.fileno(), jsb_console.fileno(), jsb.fileno()]
#        try:
#            (rin, win, xin) = select.select(rin, [], [], 1.0)
#        except select.error:
 #           util.check_parent()
#            continue

 #       tnow = time.time()

 #       if jsb_in.fileno() in rin:
 #           buf = jsb_in.recv(fdm.packet_size())
 #           process_jsb_input(buf, simtime)
 #           frame_count += 1
 #           new_frame = True

#        if sim_in.fileno() in rin:
#            simbuf = sim_in.recv(28)
#            process_sitl_input(simbuf)
#            simtime += simstep
#            last_sim_input = tnow

        # show any jsbsim console output
 #       if jsb_console.fileno() in rin:
 #           util.pexpect_drain(jsb_console)
 #       if jsb.fileno() in rin:
 #           util.pexpect_drain(jsb)

        # only simulate wind above 5 meters, to prevent crashes while
        # waiting for takeoff
  #      if tnow - last_wind_update > 0.1:
  #          update_wind(wind)
  #          last_wind_update = tnow

  #      if tnow - last_report > 3:
  #          print("FPS %u asl=%.1f agl=%.1f roll=%.1f pitch=%.1f a=(%.2f %.2f %.2f) AR=%.1f" % (
  #              frame_count / (time.time() - last_report),
  #              fdm.get('altitude', units='meters'),
  #              fdm.get('agl', units='meters'),
  #              fdm.get('phi', units='degrees'),
  #              fdm.get('theta', units='degrees'),
  #              fdm.get('A_X_pilot', units='mpss'),
  #              fdm.get('A_Y_pilot', units='mpss'),
  #              fdm.get('A_Z_pilot', units='mpss'),
  #              achieved_rate))

   #         frame_count = 0
   #         last_report = time.time()

   #     if new_frame:
   #         now = time.time()
   #         if now < last_wall_time + scaled_frame_time:
   #             dt = last_wall_time+scaled_frame_time - now
   #             time.sleep(last_wall_time+scaled_frame_time - now)
   #             now = time.time()

   #         if now > last_wall_time and now - last_wall_time < 0.1:
   #             rate = 1.0/(now - last_wall_time)
   #             achieved_rate = (0.98*achieved_rate) + (0.02*rate)
   #             if achieved_rate < opts.rate*opts.speedup:
   #                 scaled_frame_time *= 0.999
   #             else:
   #                 scaled_frame_time *= 1.001

   #         last_wall_time = now


def exit_handler():
    """Exit the sim."""
    print("running exit handler")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    out_file.close()
    # JSBSim really doesn't like to die ...
#    if getattr(jsb, 'pid', None) is not None:
#        os.kill(jsb.pid, signal.SIGKILL)
#    jsb_console.send('quit\n')
#    jsb.close(force=True)
    ##util.pexpect_close_all()
    sys.exit(1)

signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGTERM, exit_handler)

try:
    main_loop()
except Exception as ex:
    print(ex)
    exit_handler()
    raise


