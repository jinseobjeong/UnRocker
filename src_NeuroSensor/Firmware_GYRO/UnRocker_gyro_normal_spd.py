#!/usr/bin/env python

import os, signal
import subprocess
import time
import sys, argparse, math
from pymavlink import mavutil
from dronekit import connect, Command, LocationGlobal

#os.system("gazebo Tools/sitl_gazebo/worlds/iris.world")

#res = subprocess.Popen("make px4_sitl gazebo",shell=True)
#time.sleep(1000)

#subprocess.Popen("ps aux | grep gaze | awk '{print $2}' | xargs kill -9 ", shell=True)

#res.stdin.close()
#os.kill(res.pid, signal.SIGTERM)
#res.terminate()
#time.sleep(3)
#os.kill(res.pid, signal.SIGKILL)
#time.sleep(10)

DIALECT = 'custom'
global vehicle 
#global current_wp
MAV_MODE_AUTO   = 4
if (sys.version_info[0] >= 3):
    ENCODING = 'ascii'
else:
    ENCODING = None

    def check_falldown(vehicle):
#        m = vehicle._master.mav.recv_match(type='SIMSTATE', blocking=True)
#        m2 = vehicle._master.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if (vehicle.location.global_relative_frame.alt < 10.0):
            return True
#        elif (abs(math.degrees(m.roll)) > 90.0):
#            return True
        else:
            return False

    def check_goheaven(vehicle):
#        m = vehicle._master.mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if (vehicle.location.global_relative_frame.alt > 150000.0):
            return True
#        elif (abs(math.degrees(m.roll)) > 90.0):
#            return True
        else:
            return False

    # TODO: need to add more checks
    def is_abnormal(vehicle):
#        if (vehicle.mode != MAV_MODE_AUTO):
#            print("Mode Changed to LAND")
        if check_falldown(vehicle=vehicle):
            return True
        elif check_goheaven(vehicle=vehicle):
            return True
        else:
            return False

    # Checks if there exists an abnormal behavior during a roundtrip.
    # This function is based on wait_wapoint in common.py
    def check_roundtrip(vehicle,
                        out_file,
                  #      current_wp,
                  #      allow_skip=True,
                  #      max_dist=2,
                        timeout=5000):
        tstart = time.time()#self.get_sim_time()
        # this message arrives after we set the current WP
        start_wp = vehicle.commands.next#_master.mav.waypoint_current()
        current_wp = start_wp
        mode = vehicle.mode#_master.mav.flightmode
        last_wp_msg = 0
	roundtrip_start = False
	roundtrip_check = False
        while time.time() < tstart + timeout:
            if is_abnormal(vehicle=vehicle):
                print("Found abnormal behavior!")
                return True

            seq = vehicle.commands.next#self.mav.waypoint_current()
#            m = vehicle._master.mav.recv_match(type='NAV_CONTROLLER_OUTPUT', blocking=True)
#            wp_dist = m.wp_dist
#            m = vehicle._master.mav.recv_match(type='VFR_HUD', blocking=True)

            # if we changed mode, fail
         #   if vehicle.mode != mode:
         #       self.progress("WHAT? Mode changed: %s -> %s" % (mode, self.mav.flightmode))
                #out_file.write("Mode Changed to LAND\n")
            if time.time() - last_wp_msg > 1:
                ##self.progress("WP %u (wp_dist=%u Alt=%.02f), current_wp: %u,"
                ##              "wpnum_end: %u" %
                ##              (seq, wp_dist, m.alt, current_wp, wpnum_end))
                last_wp_msg = time.time()#self.get_sim_time_cached()
            if seq == current_wp+1: #or (seq > current_wp+1 and allow_skip):
                print("test: Starting new waypoint %u" % seq)
                tstart = time.time()#self.get_sim_time()
                current_wp = seq
                # the wp_dist check is a hack until we can sort out
                # the right seqnum for end of mission
            # if current_wp == wpnum_end or (current_wp == wpnum_end-1 and
            #                                wp_dist < 2):
#	    if roundtrip_start == False and seq == 3:
#		roundtrip_start = True
#	    if roundtrip_start == True and seq == 4:
#		roundtrip_check = True
#	    if roundtrip_check == True and seq ==3:
#                print("One round trip succeeded")
#		return False
            if current_wp == 7:#wpnum_end and wp_dist < max_dist:
                print("Reached final waypoint %u" % seq)
                return False
            if seq >= 255:
                print("Reached final waypoint %u" % seq)
                return False
            if seq > current_wp+1:
                print("WHAT? Skipped waypoint: %u -> %u" % (seq, current_wp+1))

        msg = 'Time-out occurred\n'
        print(msg)
        out_file.write(msg)


        return False



def run_iteration():
    res = subprocess.Popen("make px4_sitl gazebo",shell=True)
    time.sleep(10)
    ##mavproxy = start_MAVProxy_SITL()
    

    # Connect to the Vehicle
    print "Connecting"
    connection_string = '127.0.0.1:14540'
    vehicle = connect(connection_string, wait_ready=True)
    ################################################################################################
    # Listeners
    ################################################################################################

    home_position_set = False

    #Create a message listener for home position fix
    @vehicle.on_message('HOME_POSITION')
    def listener(self, name, home_position):
        global home_position_set
        home_position_set = True

    while not home_position_set:
        print "Waiting for home position..."
        time.sleep(1)

    # Display basic vehicle state
    print " Type: %s" % vehicle._vehicle_type
    print " Armed: %s" % vehicle.armed
    print " System status: %s" % vehicle.system_status.state
    print " GPS: %s" % vehicle.gps_0
    print " Alt: %s" % vehicle.location.global_relative_frame.alt

#    res2 = subprocess.Popen("mavproxy.py --master=udp:127.0.0.1:14550 --out=127.0.0.1:18570",shell=True)

    
    time.sleep(3)
    
    
    cmds = vehicle.commands
    cmds.clear()
    
    #while not vehicle.is_armable:
    #    print " Waiting for vehicle to initialise..."
    #    time.sleep(1)
        
        
    # Change to AUTO mode
    PX4setMode(MAV_MODE_AUTO)
    time.sleep(1)

    vehicle.armed= True
    home = vehicle.location.global_relative_frame
    parm_name = 'ATTACK_TRIGGER'
    time.sleep(5)

#    if vehicle._master.mavlink10():
#        print('MAVLINK 10')
#    else:
#        print('MAVLINK 20')
    vehicle.parameters['attack_trigger'] = 1.0
    print "attack : %s" %vehicle.parameters['attack_trigger']
#    vehicle._master.mav.param_set_send(vehicle._master.target_system, vehicle._master.target_component, parm_name.encode('utf8'), 1, mavutil.mavlink.MAV_PARAM_TYPE_INT16)
#    print "attack : %s" %vehicle.parameters['ATTACK_TRIGGER']
    
    time.sleep(5)
    vehicle.parameters['attack_trigger'] = 0
    print "attack : %s" %vehicle.parameters['attack_trigger']


    # takeoff to 10 meters
#    wp = get_location_offset_meters(home, 0, 0, 10);
#    cmd = Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
#    cmds.add(cmd)


#    fly_rv_fuzzer(mavproxy)
   ## fly_test(mavproxy)
    time.sleep(300)
    # usage example:
    # res = get_dynamics()
    # print("%f" % res['lat'])
 

def fly_rv_fuzzer():
#    self.progress('Start RV Fuzzer test')

#    self.context_push()

    res_fname = 'results/rv_fuzzer.txt'
    out= open(res_fname, 'w')
#    wp_fname = 'CMAC-circuit.txt'
    MAX_COUNT = 40000
    attack_amp = 2
    attack_freq = 170#245
    wp_report = 0
#    gyro_p_nse = [0.01, 0.05, 0.1]#200224
#    mavproxy.send('param set INS_ACCATK_TRG 0\n')
#    mavproxy.send('param set INS_ATTACK_TRG 0\n')
#    mavproxy.send('param set SIM_ATTACK_TRG 0\n')

        # waypoint loading
#    num_wp = self.load_mission(wp_fname)

#        # store current snapshot
#        self.fetch_parameters()
#        idx = self.mavproxy.expect(['Saved [0-9]+ parameters to (\S+)'])
#        param_fname = self.mavproxy.match.group(1)
#        self.repeatedly_apply_parameter_file(param_fname)
        # actual fuzzing starts
    for m_count in range(MAX_COUNT):
        time.sleep(10)
        res3 = subprocess.Popen("/home/sysseclab/anaconda3/envs/tf3/bin/python /home/sysseclab/prj_jsjeong/gyro_autoencoder_1D_PX4_SITL_MA/trt_inference.py",shell=True)
        time.sleep(200)

        res = subprocess.Popen("make px4_sitl gazebo",shell=True)
        time.sleep(10)
    
        # Connect to the Vehicle
        print "Connecting"
        connection_string = '127.0.0.1:14540'
        vehicle = connect(connection_string, wait_ready=True)
        ############################################################################################
        # Listeners
        ############################################################################################
        time.sleep(10)

        #####

        home_position_set = False
        vehicle.parameters['attack_trigger'] = 0
        time.sleep(1)
        vehicle.parameters['attacc_trigger'] = 0
        time.sleep(1)
        vehicle.parameters['attack_trigger'] = 1
        time.sleep(3)
        vehicle.parameters['attack_trigger'] = 0
        time.sleep(1)

        #Create a message listener for home position fix
#        @vehicle.on_message('HOME_POSITION')
#        def listener(self, name, home_position):
#            global home_position_set
#            home_position_set = True

#        while not home_position_set:
#            print "Waiting for home position..."
#            time.sleep(1)
        # Display basic vehicle state
        print " Type: %s" % vehicle._vehicle_type
        print " Armed: %s" % vehicle.armed
        print " System status: %s" % vehicle.system_status.state
        print " GPS: %s" % vehicle.gps_0
        print " Alt: %s" % vehicle.location.global_relative_frame.alt
#    res2 = subprocess.Popen("mavproxy.py --master=udp:127.0.0.1:14550 --out=127.0.0.1:18570",shell=True)
        time.sleep(3)
    
    
#        cmds = vehicle.commands
#        cmds.clear()
    
        
        # Change to AUTO mode
#        PX4setMode(MAV_MODE_AUTO)
        vehicle._master.mav.command_long_send(vehicle._master.target_system, 
                vehicle._master.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 
                0, MAV_MODE_AUTO, 0, 0, 0, 0, 0, 0)

        time.sleep(1)

        vehicle.armed= True
        home = vehicle.location.global_relative_frame

        time.sleep(35)
        vehicle.parameters['attack_amp'] = attack_amp
        vehicle.parameters['attack_freq'] = attack_freq
        vehicle.parameters['attack_trigger'] = 1

        print "attack : %s" %vehicle.parameters['attack_trigger']

       
        crashed = check_roundtrip(vehicle=vehicle, out_file = out)#, current_wp = wp_report)
        if crashed:
                #msg = 'Crashed INS_All ATTACK at CUTOFF 5, AMP: %f, FREQ: %d\n' % (attack_amp,
#                                                                        attack_freq)
            wp_report = vehicle.commands.next
            msg = 'Crashed SIM_gyro ATTACK at CUTOFF 20, D 0.5, AMP: %f, FREQ: %d, WP %d\n' % (attack_amp, attack_freq, wp_report)
            out.write(msg)
            attack_amp = 1.5#temporary 191020
            attack_freq += 5#2#0.1
               # if (attack_freq%5) == 0:
               #     attack_freq += 2

        else:
                #msg = 'Not crashed INS_All ATTACK at CUTOFF 5, AMP: %f, FREQ: %d\n' % (attack_amp,
                                                                 #        attack_freq)
            msg = 'Not crashed SIM_gyro ATTACK at CUTOFF 20, D 0.5, AMP: %f, FREQ: %d\n' % (attack_amp, attack_freq)
 
            out.write(msg)
            attack_amp += 0.5
        if attack_amp > 2.0:
#                msg = 'Crashed INS_All ATTACK at CUTOFF 5, AMP: 4.500000, FREQ: %d\n' % (attack_freq)
            msg = 'Crashed SIM_gyro ATTACK at CUTOFF 20, D 0.5, AMP: 4.5, FREQ: %d\n' % (attack_freq)
 
            out.write(msg)
            attack_amp = 1.5
            attack_freq += 5
            #    if (attack_freq%5) == 0:
            #        attack_freq += 2


#        self.progress(msg)
            #self.progress(msg, res_fname)
            #attack_amp += 0.01
            #if attack_amp > 0.15:
            #    attack_freq += 0.1
        vehicle.parameters['attack_trigger'] = 0
        time.sleep(1)
        vehicle.parameters['attack_trigger'] = 0
        time.sleep(1)


        subprocess.Popen("ps aux | grep gaze | awk '{print $2}' | xargs kill -9 ", shell=True)
        subprocess.Popen("ps aux | grep gzcli | awk '{print $2}' | xargs kill -9 ", shell=True)
        subprocess.Popen("ps aux | grep trt_inference | awk '{print $2}' | xargs kill -9 ", shell=True)



#        self.zero_throttle()
#        self.disarm_vehicle(force=True)
#        self.mavproxy.send('param set INS_ACCATK_TRG 0\n')
#        self.mavproxy.send('param set INS_ATTACK_TRG 0\n')
#        self.mavproxy.send('param set SIM_ATTACK_TRG 0\n')

 #       self.reboot_sitl()
##        if attack_freq > 5:
##            attack_freq = 245
        if attack_freq > 170:
            break
    out.close()
    self.progress("RVFuzzer test is done.")

def start_MAVProxy_SITL(master='udp:127.0.0.1:14550'):#:5760'):##udp
    """Launch mavproxy connected to a SITL instance."""
    import pexpect
    #global close_list
    MAVPROXY = os.getenv('MAVPROXY_CMD', 'mavproxy.py')
    cmd = MAVPROXY + ' --master=%s --out=127.0.0.1:18570' % master##14540
   # if setup:
   #     cmd += ' --setup'
#    if aircraft is None:
#        aircraft = 'test.%s' % atype
#    cmd += ' --aircraft=%s' % aircraft
#    cmd += ' ' + ' '.join(options)
#    cmd += ' --default-modules misc,terrain,wp,rally,fence,param,arm,mode,rc,cmdlong,output'
    ret = pexpect.spawn("mavproxy.py --master=udp:127.0.0.1:14550 --out=127.0.0.1:1857", encoding=ENCODING, timeout=60)#cmd, encoding=ENCODING, timeout=60)
    ret.delaybeforesend = 0
#    pexpect_autoclose(ret)
    return ret


def tests(self):
    return [
        ("RVFuzzer Test",
             "Test Fly RV Fuzzer Tests (takeoff)",
         self.fly_rv_fuzzer),
    ]

def fly_test(mavproxy):
    print("test")
    mavproxy.send('commander takeoff\n')

#def PX4setMode(mavMode):
#    vehicle._master.mav.command_long_send(vehicle._master.target_system, vehicle._master.target_component,
#                                               mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
#                                               mavMode,
#                                               0, 0, 0, 0, 0, 0)

def get_location_offset_meters(original_location, dNorth, dEast, alt):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned Location adds the entered `alt` value to the altitude of the `original_location`.
    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon,original_location.alt+alt)



if __name__ == "__main__":
#    fly_test()
#    res = subprocess.Popen("make px4_sitl gazebo",shell=True)
#    time.sleep(10)
    ##mavproxy = start_MAVProxy_SITL()
    

    # Connect to the Vehicle
#    print "Connecting"
#    connection_string = '127.0.0.1:14540'
#    vehicle = connect(connection_string, wait_ready=True)
    ################################################################################################
    # Listeners
    ################################################################################################

#    home_position_set = False

    #Create a message listener for home position fix
#    @vehicle.on_message('HOME_POSITION')
#    def listener(self, name, home_position):
#        global home_position_set
#        home_position_set = True

#    while not home_position_set:
#        print "Waiting for home position..."
#        time.sleep(1)
    # Display basic vehicle state
#    print " Type: %s" % vehicle._vehicle_type
#    print " Armed: %s" % vehicle.armed
#    print " System status: %s" % vehicle.system_status.state
#    print " GPS: %s" % vehicle.gps_0
#    print " Alt: %s" % vehicle.location.global_relative_frame.alt
#    res2 = subprocess.Popen("mavproxy.py --master=udp:127.0.0.1:14550 --out=127.0.0.1:18570",shell=True)
 #   time.sleep(3)
    
    
#    cmds = vehicle.commands
#    cmds.clear()
    
    #while not vehicle.is_armable:
    #    print " Waiting for vehicle to initialise..."
    #    time.sleep(1)
        
        
    # Change to AUTO mode
#    PX4setMode(MAV_MODE_AUTO)
#    time.sleep(1)

 #   vehicle.armed= True
#    home = vehicle.location.global_relative_frame
#    parm_name = 'ATTACK_TRIGGER'
#    time.sleep(10)

#    if vehicle._master.mavlink10():
#        print('MAVLINK 10')
#    else:
#        print('MAVLINK 20')
#    vehicle.parameters['attack_trigger'] = 1.0
#    print "attack : %s" %vehicle.parameters['attack_trigger']
#    vehicle._master.mav.param_set_send(vehicle._master.target_system, vehicle._master.target_component, parm_name.encode('utf8'), 1, mavutil.mavlink.MAV_PARAM_TYPE_INT16)
#    print "attack : %s" %vehicle.parameters['ATTACK_TRIGGER']
    
#    time.sleep(20)
#    vehicle.parameters['attack_trigger'] = 0
#    print "attack : %s" %vehicle.parameters['attack_trigger']


    # takeoff to 10 meters
#    wp = get_location_offset_meters(home, 0, 0, 10);
#    cmd = Command(0,0,0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 1, 0, 0, 0, 0, wp.lat, wp.lon, wp.alt)
#    cmds.add(cmd)


#    fly_rv_fuzzer(mavproxy)
   ## fly_test(mavproxy)
#    time.sleep(30)
    fly_rv_fuzzer()
#    subprocess.Popen("ps aux | grep gaze | awk '{print $2}' | xargs kill -9 ", shell=True)
#    subprocess.Popen("ps aux | grep gzcli | awk '{print $2}' | xargs kill -9 ", shell=True)

    # usage example:
    # res = get_dynamics()
    # print("%f" % res['lat'])
 
