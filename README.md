# UnRocking Drones : Foundattions of Acoustic Injection Attacks and Recovery Thereof (NDSS2023)

### Requirements 

Requirements are specified in "requirements.txt"

**$pip install -r requirements.txt**

---

## Drone firmware 
#### Our testbed is based on industry-leading opensource PX4 drone firmware. 
#### Modified codes for acoustic injection tests 
 - Resonant sensor models are implanted in sensor driver.
 - Attack parameters (induced frequency and attack amplitude) are connected to external commands.
 - Mavlink module was modified to interface with the external commands.
 
#### Modified codes for implication analysis 
 - Additional logs : additional drone state variables, hardware timings are included.

#### Misc 
 - ROMFS (ROM Filesystem) file was revised to commnicate with inference computer.
 
---

## Automation Script 

#### Iterative automated testing python code 

### Software-In-The-Loop(SITL) Automated Testing 

 - It provides software only testing.

**$cd UnRocker_FW**

**$./UnRocker_Gyro_SITL.py** or **$./UnRocker_Accel_SITL.py**

### Hardware-In-The-Loop(HITL) Automated Testing 
 - Hardware related testing, so it requires the flight controller (FC).

**$cd UnRocker_FW**

**$./UnRocker_Gyro_HITL.py** or **$./UnRocker_Accel_HITL.py**

### Automated Dataset Generation (HITL) 
 - Automated Dataset Generation is based on HITL mode.

**$cd UnRocker_FW**

**$./UnRocker_TestsetGen_HITL.py** 


---

## UnRocker recovery 
#### DAE design training 
 - Denoising AutoEncoder (DAE) is our core network.

**$cd UnRocker_DAE_Gyro** or **$cd UnRokcer_DAE_Accel**

**$python3 train_conv_autoeoncoder.py**

#### Dataset 
 - Automatically generated dataset from HITL.

#### Test Examples 
 - Basic testset : Quadcopter Iris and Solo.
 - Drone flight test data
 - Actual injection test data

**$cd UnRocker_DAE_Gyro** or **$cd  UnRokcer_DAE_Accel**

**$python3 train_conv_autoeoncoder.py**


#### Realtime Inference 
 - Realtime inference code based on TensorRT

**$cd UnRocker_DAE_Gyro** or **$cd UnRokcer_DAE_Accel**

**$python3 trt_inference.py**

---

## Reference
### Citation

@inproceedings{jeong2023unrocker,
  title={UnRocking Drones : Foundattions of Acoustic Injection Attacks and Recovery Thereof},
  author={Jeong, Jinseob and Kim, Dongkwan and Jang, Joonha and Noh, Juhwan and Song, Changhun and Kim, Yongdae},
  booktitle={Network and Distributed Systems Security (NDSS) Symposium},
  year={2023}
}

### Webpage
**https://sites.google.com/view/unrocker/**

### Contact
**e-mail: jeongjinseob@gmail.com**
