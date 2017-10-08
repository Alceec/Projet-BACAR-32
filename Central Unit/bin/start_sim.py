#! /usr/bin/env python

import subprocess
import time
import sys
import argparse

# CHANGE THE FOLLOWING LINE IF YOU WANT TO CHANGE THE DEFAULT arena
DEFAULT_ARENA = '../data/arena_manhattan.json'

# CHANGE THE FOLLOWING LINES IF YOU WANT TO EXECUTE DIFFERENT MODULES
STATEMACHINE_MODULE = 'state_machine'
PATH_MODULE = 'path_detector'
SIGN_MODULE = 'sign_detector'

# NO CHANGES BELOW THIS LINE ARE NECESSARY

PYTHON2 = 'python'
PYTHON3 = 'python3'


def get_arguments():
    parser = argparse.ArgumentParser(description='%s: Launch simulator, path detector, sign detector, and state machine' % __file__)
    parser.add_argument('--arena', help='the simulated arena to use (DEFAULT=%s)' % DEFAULT_ARENA, default=DEFAULT_ARENA)
    return parser.parse_args()


args = get_arguments()


statemachine_cmd = [PYTHON3, '../bin/sm_server.py', '--machine', STATEMACHINE_MODULE]
# path_cmd = [PYTHON3, '../bin/path_server.py', '--detector', PATH_MODULE, '--display']
path_cmd = [PYTHON3, '../bin/path_server.py', '--detector', PATH_MODULE]
# sign_cmd = [PYTHON3, '../bin/sign_server.py', '--detector', SIGN_MODULE, '--display']
sign_cmd = [PYTHON3, '../bin/sign_server.py', '--detector', SIGN_MODULE]
viewer_cmd = [PYTHON2, '../bin_remote/viewer.py', '--local']
simulator_cmd = [PYTHON2, '../bin/simulator.py', '--arena', args.arena]  # includes traffic light detector

commands = [statemachine_cmd, path_cmd, sign_cmd, viewer_cmd, simulator_cmd]
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
