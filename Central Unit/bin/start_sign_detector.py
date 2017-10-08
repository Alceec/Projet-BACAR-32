#! /usr/bin/env python
import subprocess
import time
import sys
import argparse

# CHANGE THE FOLLOWING LINES IF YOU WANT TO EXECUTE DIFFERENT MODULES

SIGN_MODULE = 'sign_detector' # = ./sign_detector.py

# NO CHANGES BELOW THIS LINE ARE NECESSARY


def get_arguments():
    parser = argparse.ArgumentParser(description='Start Sign Detector. Launches IMAGE SERVER using the camera; and SIGN DETECTOR, and STATE MACHINE.')
    return parser.parse_args()


args = get_arguments()

PYTHON2 = 'python'
PYTHON3 = 'python3'

sign_cmd = [PYTHON3, '../bin/sign_server.py', '--detector', SIGN_MODULE]
viewer_cmd = [PYTHON2, '../bin_remote/viewer.py', '--local']
imageserver_cmd = [PYTHON2, '../bin/image_server.py', '--filter', '--show']

commands = [sign_cmd, viewer_cmd, imageserver_cmd]
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
