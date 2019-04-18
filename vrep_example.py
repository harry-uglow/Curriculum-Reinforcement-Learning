####################################################################################################################
# Example code for controlling V-Rep using the remote API
# This is based on the old API (Legacy API) rather than the new API (BlueZero API)
# Read here for the Legacy API: http://www.coppeliarobotics.com/helpFiles/en/legacyRemoteApiOverview.htm
# This script may need to be changed if you want to use it with the latest version of V-Rep using the BlueZero API
# Make sure that "example_scene.ttt" is in the same directory as this script
# If you run this script, it should launch V-Rep, load the example scene, and then beging moving the robot
#####################################################################################################################


# Import some modules
import os
import signal
import time
from subprocess import Popen
import numpy as np
import vrep

np.set_printoptions(precision=2, linewidth=200)


# Function to check for errors when calling a remote API function
def check_for_errors(code):
    if code == vrep.simx_return_ok:
        return
    elif code == vrep.simx_return_novalue_flag:
        # Often, not really an error, so just ignore
        pass
    elif code == vrep.simx_return_timeout_flag:
        raise RuntimeError('The function timed out (probably the network is down or too slow)')
    elif code == vrep.simx_return_illegal_opmode_flag:
        raise RuntimeError('The specified operation mode is not supported for the given function')
    elif code == vrep.simx_return_remote_error_flag:
        raise RuntimeError('The function caused an error on the server side (e.g. an invalid handle was specified)')
    elif code == vrep.simx_return_split_progress_flag:
        raise RuntimeError('The communication thread is still processing previous split command of the same type')
    elif code == vrep.simx_return_local_error_flag:
        raise RuntimeError('The function caused an error on the client side')
    elif code == vrep.simx_return_initialize_error_flag:
        raise RuntimeError('A connection to vrep has not been made yet. Have you called connect()? (Port num = ' + str(port_num))


# Function to call a Lua function in V-Rep
# Some things (such as getting the robot's joint velocities) do not have a remote (Python) API function, only a regular API function
# Therefore, they need to be called directly in V-Rep using Lua (see the script attached to the "remote_api" dummy in the V-Rep scene)
# Read more here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiExtension.htm
def call_lua_function(lua_function, ints=[], floats=[], strings=[], bytes=bytearray(), opmode=vrep.simx_opmode_blocking):
    return_code, out_ints, out_floats, out_strings, out_buffer = vrep.simxCallScriptFunction(client_id, 'remote_api', vrep.sim_scripttype_customizationscript, lua_function, ints, floats, strings, bytes, opmode)
    check_for_errors(return_code)
    return out_ints, out_floats, out_strings, out_buffer


# Define the port number where communication will be made to the V-Rep server
port_num = 19990
# Define the host where this communication is taking place (the local machine, in this case)
host = '127.0.0.1'

print("Launching vrep")
# Launch a V-Rep server
# Read more here: http://www.coppeliarobotics.com/helpFiles/en/commandLine.htm
remote_api_string = '-gREMOTEAPISERVERSERVICE_' + str(port_num) + '_FALSE_TRUE'
args = ['/homes/hu115/Desktop/V-REP_PRO_EDU_V3_6_0_Ubuntu18_04/vrep.sh', remote_api_string]
process = Popen(args, preexec_fn=os.setsid)
time.sleep(6)

print("Starting Connection")

# Start a communication thread with V-Rep
client_id = vrep.simxStart(host, port_num, True, True, 5000, 5)
return_code = vrep.simxSynchronous(client_id, enable=True)
check_for_errors(return_code)

print("Loading Scene")

# Load the scene
dir_path = os.path.dirname(os.path.realpath(__file__))
scene_path = dir_path + '/reacher.ttt'
return_code = vrep.simxLoadScene(client_id, scene_path, 0, vrep.simx_opmode_blocking)
check_for_errors(return_code)

print("Getting config")

# Get the initial configuration of the robot (needed to later reset the robot's pose)
init_config_tree, _, _, _ = call_lua_function('get_configuration_tree', opmode=vrep.simx_opmode_blocking)
_, init_joint_angles, _, _ = call_lua_function('get_joint_angles', opmode=vrep.simx_opmode_blocking)

print("Getting joint angles")

# Get V-Rep handles for the robot's joints
joint_handles = [None] * 7
for i in range(7):
    return_code, handle = vrep.simxGetObjectHandle(client_id, 'Sawyer_joint' + str(i + 1), vrep.simx_opmode_blocking)
    check_for_errors(return_code)
    joint_handles[i] = handle

print("Starting")

# Start the simulation (the "Play" button in V-Rep should now be in a "Pressed" state)
return_code = vrep.simxStartSimulation(client_id, vrep.simx_opmode_blocking)
check_for_errors(return_code)

# Loop over episodes
for episode_num in range(100):

    # At the beginning of each episode, reset the robot to its initial position
    # When the joints are in torque/force mode, this needs to be done using the regular API (i.e. calling a Lua function in the V-Rep scene)
    call_lua_function('set_joint_angles', ints=init_config_tree, floats=init_joint_angles)

    # Define the initial velocities for the robot's joints
    target_velocities = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Loop over steps in this episode
    for step_num in range(100):

        # Randomly change the joint velocities
        # Need to pause and unpause communication before and after (see "simxPauseCommunication" here: http://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctionsMatlab.htm)
        vrep.simxPauseCommunication(client_id, 1)
        for i in range(7):
            target_velocities[i] += np.random.uniform(-0.01, 0.01)
        vrep.simxPauseCommunication(client_id, 0)
        print('Target velocities = ' + str(target_velocities))

        # Set the target joint velocities
        for i in range(7):
            return_code = vrep.simxSetJointTargetVelocity(client_id, joint_handles[i], target_velocities[i], vrep.simx_opmode_oneshot)
            check_for_errors(return_code)

        # Get the actual velocities of the robot's joints
        _, actual_velocities, _, _ = call_lua_function('get_joint_velocities', ints=joint_handles, opmode=vrep.simx_opmode_blocking)
        print('Actual velocities = ' + str(np.array(actual_velocities)))

        # Do one step of the simulation (both these lines are needed)
        vrep.simxSynchronousTrigger(client_id)
        vrep.simxGetPingTime(client_id)

# Shutdown
vrep.simxStopSimulation(client_id, vrep.simx_opmode_blocking)
vrep.simxFinish(client_id)
pgrp = os.getpgid(process.pid)
os.killpg(pgrp, signal.SIGINT)
