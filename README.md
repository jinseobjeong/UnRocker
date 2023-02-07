# UnRocking Drones : Foundattions of Acoustic Injection Attacks and Recovery Thereof (NDSS2023)

### Requirements ###
Requirements are specified in `requirements.txt'
$pip install -re requirements.txt

## Drone Firmware ##
Our testbed is baed on industry-leading opensource PX4 drone firmware.
### Modified Codes for Acoustic Injection Tests ###
Resonant sensor models are implanted in sensor driver.
Attack parameters (induced frequency and attack amplitude) are connected to external commands.
Mavlink module was modified to interface the external commands.
### Modified Codes for Implication Analysis ###
Additional logs : additional drone state variables, hardware timings.
### Misc ###
ROMFS (ROM Filesystem) file was revised to commnicate with inference computer.

## Automation Script ##
Iterative automated testing python code
### Software-In-The-Loop(SITL) Automated Testing ###
Software only testing.
### Hardware-In-The-Loop(HITL) Automated Testing ###
Hardware related testing, so it requires the flight controller (FC).
### Automated Dataset Generation (HITL) ###
Automated Dataset Generation is based on HITL mode.

## UnRocker Recovery ##
### DAE design Training ###
Denoising AutoEncoder (DAE) is our core network.
### Dataset ###
Automatically generated dataset from HITL.
### Test Examples ###
Basic Testset (IRIS/SOLO)
Drone Flight Testset
Actual Injection Testset
### Real-time Inference ###
Real-time inference code based on TensorRT

