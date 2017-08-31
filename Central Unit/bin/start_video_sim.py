#! /usr/bin/env python
import subprocess
import time
import sys
import argparse

# CHANGE THE FOLLOWING LINES IF YOU WANT TO EXECUTE DIFFERENT MODULES

STATEMACHINE_MODULE = 'state_machine' # = ./state_machine.py
PATH_MODULE = 'path_detector' # = ./path_detector.py
SIGN_MODULE = 'sign_detector' # = ./sign_detector.py

# NO CHANGES BELOW THIS LINE ARE NECESSARY


def get_arguments():
    parser = argparse.ArgumentParser(description='Start Video Simulation: launch image server with video file; path detector, sign detector, and state machine')
    parser.add_argument('video_file', help='specify the video file to play')
    return parser.parse_args()


args = get_arguments()

PYTHON = 'python'

statemachine_cmd = [PYTHON, '../bin/sm_server.py', '--machine', STATEMACHINE_MODULE]
path_cmd = [PYTHON, '../bin/path_server.py', '--detector', PATH_MODULE, '--display']
sign_cmd = [PYTHON, '../bin/sign_server.py', '--detector', SIGN_MODULE, '--display']
imageserver_cmd = [PYTHON, '../bin/image_server.py', '--bird', '--vflip', '--filter', '--show', args.video_file]

commands = [statemachine_cmd, path_cmd, sign_cmd, imageserver_cmd]
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
