#!/usr/bin/python

"""
Listen to MQTT messages sent by the image server (in particular: the plausible
signs) and emit MQTT messages that emit the type of sign detected

Arguments:
   --detector:  the detector module to use. This is a python module that has a
                 SignDetector class which has a single detect() method
                 that gets the sign as input (a numpy array) as well as its bounding
                 box in the source image and returns
                 the type of sign detected. The default detector is specified
                 in the config_server config file.

    --config: config file to use (default config_server.json)

    --display: whether to display the recognized sign information,
               usefull for debugging (default = off)

    --profile: whether to profile the program and display results on
               exit (default = off)


INPUT MQTT Events:

mqttconfig.MSG_SERVER_SIGN_ARRAY: the sign array produced by the image server (includes bounding box)

OUTPUT MQTT events:

mqttconfig.MSG_ANALYZER_SIGN_JSON: the dict produced produced by the analyzer  (includes type + distance)

"""

import cProfile
import pstats
import paho.mqtt.client as mqtt
import logging
import json
import time
import argparse
import importlib
import cv2
import numpy as np
import types

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from os import sys, path

# add directory containing current script to search path
sys.path.append(path.abspath(path.dirname(__file__)+'/.'))
# add parent directory of current script to search path
sys.path.append(path.abspath(path.dirname(__file__)+'/..'))

# add work directory (i.e., the directory from which the script is called) to search path
sys.path.append(path.abspath('.'))

import six
from tools.param_tools import read_param
from tools.log_tools import create_logger
from tools.fps_tools import fps_generator
from tools.compression_tools import payload_to_ts_bb_signarray

from mqtt_config import *


class Data:
    def __init__(self, parameters):
        self.fps = fps_generator(maxlen=10)
        self.current_fps = 0.0
        self.detector = None
        self.params = parameters


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MSG_SERVER_SIGN_ARRAY, qos=0)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # userdata contains an object of type Data

    # process incoming boolean array that contains the corrected camera image
    if msg.topic == MSG_SERVER_SIGN_ARRAY:
        userdata.current_fps = six.next(userdata.fps)
        # convert binary payload of the incoming message to mask timestamp and mask
        (sign_ts, (x0, y0, w, h), sign_arr) = payload_to_ts_bb_signarray(msg.payload)
        if userdata.detector is not None:
            # call user-defined detector to retrieve:
            # - a dictionary that describes the sign detected (e.g., type,
            #    estimated distance, ...)
            sign_dict = userdata.detector.detect((x0, y0, w, h), sign_arr)

            if sign_dict is None:
                logging.warning("Detector returned None for path_dict")
            else:
                if type(sign_dict) is not dict:
                    logging.error("Detector return other kind of object than dictionary for detected sign")
                    sys.exit(-1)

                # add the timestamp of when the sign was detected, as well as the
                # timestamp of the sign image that it computed the detection information
                # for
                sign_dict['ts'] = time.time()
                sign_dict['ref_ts'] = sign_ts
                sign_json = json.dumps(sign_dict)
                # logging.info("Detection returned: %s", sign_json)
                client.publish(MSG_ANALYZER_SIGN_JSON, sign_json, qos=0)

            if userdata.params['display']:
                display(sign_arr, sign_dict, userdata)


def display(sign_arr, sign_dict, userdata):
    DEFAULTW, DEFAULTH = 100, 200
    (h, w, l) = sign_arr.shape
    canvas = np.zeros((DEFAULTH, DEFAULTW + 250, 3), np.uint8)
    canvas[0:h, 0:w, :] = sign_arr

    # text
    s = 'fps:%.1f' % (userdata.current_fps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, s, (DEFAULTW + 1, 21), font, .5, (20,20,20),1)
    cv2.putText(canvas, s, (DEFAULTW + 0, 20), font, .5, (200,200,200), 1)

    if sign_dict is not None:
        # print other keys from json dict
        y = 21  # vertical offset to start at
        for key, value in six.iteritems(sign_dict):
            y = y + 20
            # text
            if type(value) == float:
                s = '%s:%.2f' % (str(key)[0:min(len(str(key)), 10)], float(value))
            else:
                s = '%s:%s' % (str(key)[0:min(len(str(key)), 10)], str(value))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, s, (DEFAULTW + 1, y), font, .5, (20,20,20),1)
            cv2.putText(canvas, s, (DEFAULTW + 0, y-1), font, .5, (200,200,200), 1)

    cv2.imshow('Sign Detector', canvas)


def setup_logging():
    log_file_name = "./%s.log" % path.basename(__file__)
    scriptname = "%s" % path.basename(__file__)
    print("Logging to " + path.abspath(log_file_name))
    create_logger(log_file_name, scriptname=scriptname)
    # output python version
    logging.info("Running on python version" + sys.version.replace("\n", "\t"))


def get_arguments():
    parser = argparse.ArgumentParser(description='Path Server')
    parser.add_argument('--detector', nargs='?', default=None,
                        help='specify the name of the sign detector module with SignDetector class. (default: specified in config file; key = sign_detector)')
    parser.add_argument('--config', nargs='?', default=None,
                        help='specify config file to use (default: config_server.json)')
    parser.add_argument('--display', action='store_true', default=False,
                        help='whether to visualize the computed sign information. Useful for debugging (default=off)')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='profile the performance of path server and display this on exit (default=off)')

    return parser.parse_args()


def get_parameters(args):
    paramsfile = 'config_server.json'
    if args.config is not None:
        paramsfile = args.config

    # default parameters
    default_params = {'mqtt_host': 'localhost',
                      'mqtt_port': 1883,
                      'sign_detector': None,
                      'display' : False,
                      'profile' : False }
    params = read_param(paramsfile)
    if params is None:
        params = default_params
        logging.warning("Default parameters will be used")
    else:
        # config file overrides defaults
        p = default_params.copy()
        p.update(params)
        params = p
    if args.display:
        params['display'] = args.display # command line overrides config file
    if args.detector is not None:
        params['sign_detector'] = args.detector # command line overrides config file
    if args.profile:
        params['profile'] = args.profile
    return params


def get_detector(params):
    # load detector
    if params['sign_detector'] is None:
        logging.error('No detector module specified on command line or config file')
        sys.exit(-1)

    logging.info('Using detector module: %s' % params['sign_detector'])
    mod = importlib.import_module(params['sign_detector'])
    # return the module
    return mod


def setup_display(params):
    if params['display']:
        cv2.namedWindow('Sign Detector')
        cv2.moveWindow('Sign Detector',620 , 300)


if __name__ == '__main__':
    setup_logging()
    args = get_arguments()

    # read parameters file
    parameters = get_parameters(args)

    # setup userdata object (needed in on_message handler)
    userdata = Data(parameters)

    # configure detector to be used
    userdata.detector = get_detector(parameters)


    # start display if necessary
    setup_display(parameters)

    # create mqtt client -
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.user_data_set(userdata)

    logging.info('connecting to the broker: %s:%d'%(parameters['mqtt_host'], parameters['mqtt_port']))
    client.connect(parameters['mqtt_host'], parameters['mqtt_port'], 10)

    # start profiling
    if parameters['profile']:
        logging.info('start profiling')
        pr = cProfile.Profile()
        pr.enable()

    try:
        while True:
            # Wait for key (needed to display image)
            k = cv2.waitKey(1)  # waits 1 milisecond
            client.loop(0.05)
            if k & 0xFF == ord('q'):  # break the loop if "q" is keyed
                break
    except KeyboardInterrupt:
        pass
    except:
        logging.exception("Exception thrown:")

    # Display a summary of the time used by each function
    # probably most of the time is spent waiting between acquisitions
    if parameters['profile']:
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        logging.info('Profiling data ...\n'+s.getvalue())
