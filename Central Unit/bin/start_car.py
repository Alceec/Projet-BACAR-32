#! /usr/bin/env python
import subprocess
import time
import sys

# CHANGE THE FOLLOWING LINES IF YOU WANT TO EXECUTE DIFFERENT MODULES

STATEMACHINE_MODULE = 'state_machine'  # = ./state_machine.py
PATH_MODULE = 'path_detector'  # = ./path_detector.py
SIGN_MODULE = 'sign_detector' # = ./sign_detector.py

# NO CHANGES BELOW THIS LINE ARE NECESSARY

PYTHON = 'python'

statemachine_cmd = [PYTHON, 'sm_server.py', '--machine', STATEMACHINE_MODULE]
path_cmd = [PYTHON, 'path_server.py', '--detector', PATH_MODULE, '--display']
sign_cmd = [PYTHON, 'sign_server.py', '--detector', SIGN_MODULE, '--display']
imageserver_cmd = [PYTHON, 'image_server.py', '--bird', '--hide', '--filter']
driver_cmd = [PYTHON, 'driver.py']

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
