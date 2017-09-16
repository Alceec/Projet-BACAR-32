#! /usr/bin/env python
import subprocess
import time
import sys
import argparse

# CHANGE THE FOLLOWING LINES IF YOU WANT TO EXECUTE DIFFERENT MODULES

STATEMACHINE_MODULE = 'state_machine'  # = ./state_machine.py
PATH_MODULE = 'path_detector'  # = ./path_detector.py
SIGN_MODULE = 'sign_detector' # = ./sign_detector.py

# NO CHANGES BELOW THIS LINE ARE NECESSARY

PYTHON2 = 'python'
PYTHON3 = 'python3'


def get_arguments():
    parser = argparse.ArgumentParser(description='%s: Launch all required programs to drive the BAcar. Starts IMAGE SERVER (in birdsview mode, hidden), PATH DETECTOR, SIGN DETECTOR, TRAFFIC LIGHT DETECTOR, STATE MACHINE, and DRIVER' % __file__)
    return parser.parse_args()


args = get_arguments()


statemachine_cmd = [PYTHON3, '../bin/sm_server.py', '--machine', STATEMACHINE_MODULE]
path_cmd = [PYTHON3, '../bin/path_server.py', '--detector', PATH_MODULE]
sign_cmd = [PYTHON3, '../bin/sign_server.py', '--detector', SIGN_MODULE]
imageserver_cmd = [PYTHON2, '../bin/image_server.py', '--bird', '--hide', '--filter'] #includes traffic light detector
driver_cmd = [PYTHON2, '../bin/driver.py']

commands = [statemachine_cmd, path_cmd, sign_cmd, imageserver_cmd, driver_cmd]
subprocs = [subprocess.Popen(cmd) for cmd in commands]


def terminate():
    for p in subprocs:
        if p.poll() is None:
            p.kill()
    sys.exit(1)


try:
    while True:
        for p in subprocs:
            if p.poll() is not None:
                terminate()
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Quitting. Terminating all processes\n")
    terminate()
    sys.exit(0)
