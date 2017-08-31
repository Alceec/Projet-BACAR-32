#!/usr/bin/python

"""
Listen to MQTT messages sent by the image server (in particular: the boolean mask)
and emit MQTT messages that indicate path information.

Arguments:
    --detector:  the detector module to use. This is a python module that has a
                 detect() method which gets the mask as input and returns a dict
                 with path information. The default detector is specified in the
                 config_server config file (key path_detector).

    --config: config file to use (default: config_server.json)

    --display: whether to display the computed path information (default = off)

    --profile: whether to profile the program and display results on exit (default = off)


INPUT MQTT Events:

mqttconfig.MSG_BOOL_ARRAY: the maks array produced by the image server

OUTPUT MQTT events:

mqttconfig.MSG_PATH_JSON: the dict produced by the detector routine
mqttconfig.MSG_PATH_IMG:  the image produced by the detector routine
                          for display in the viewer
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
sys.path.append(path.dirname(path.abspath(__file__))+'/.')
# add parent directory of current script to search path
sys.path.append(path.dirname(path.abspath(__file__))+'/..')

# add work directory (i.e., the directory from which the script is called) to search path
sys.path.append(path.abspath('.'))

import six
from tools.param_tools import read_param,save_param
from tools.log_tools import create_logger
from tools.fps_tools import fps_generator
from tools.compression_tools import payload_to_ts_mask, ts_ref_rgb_to_payload

from mqtt_config import *


class Data:
    def __init__(self, parameters):
        self.mask = None
        self.mask_ts = None # timestamp id when mask was generated
        self.mask_recv_ts = None #timestamp when mask was received
        self.fps = fps_generator(maxlen=10)
        self.current_fps = 0.0
        self.detector = None
        self.params = parameters


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MSG_BOOL_ARRAY, qos=0)

    # TODO IS THIS STILL NECESSARY?
    client.subscribe(CMD_SET_XY, qos=0)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # userdata contains an object of type Data

    # process incoming boolean array that contains the corrected camera image
    if msg.topic == MSG_BOOL_ARRAY:
        userdata.current_fps = six.next(userdata.fps)
        #c onvert binary payload of the incoming message to mask timestamp and mask
        (mask_ts, mask) = payload_to_ts_mask(msg.payload)
        mask = mask * 255     # convert "on" pixels to white
        if userdata.detector is not None:
            # call user-defined detector to retrieve:
            # - a dictionary that describes the path detected (e.g., distance, ...)
            # - a one-channel numpy image that can be used to visualize the path (for debugging)
            (path_dict, path_img) = userdata.detector.detect(mask)

            if path_dict is None:
                logging.info("Detector returned None for path_dict")
                return

            if type(path_dict) is not dict:
                logging.error("Detector return other kind of object than dictionary for path_dict")
                sys.exit(-1)

            # add the timestamp of when the path was generated, as well as the
            # timestamp of the mask that it computed the path for
            path_dict['ts'] = time.time()
            path_dict['mask_ts'] = mask_ts
            path_json = json.dumps(path_dict)
            # logging.info("Detection returned: %s", path_json)
            client.publish(MSG_PATH_JSON, path_json, qos=0)
            if path_img is not None:
                ts = path_dict['ts']
                mask_ts = path_dict['mask_ts']
                payload = ts_ref_rgb_to_payload(ts, mask_ts, path_img)
                client.publish(MSG_PATH_IMG, payload, qos=0)

            if userdata.params['display']:
                display(mask, path_img, path_dict, userdata)


def display(mask, path_img, path_dict, userdata):
    # assert(mask.shape == path_img.shape)
    (h, w) = mask.shape
    canvas = np.zeros((h, w*2, 3), np.uint8)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if path_img is not None:
        path_img_mask = cv2.threshold(cv2.cvtColor(path_img, cv2.COLOR_BGR2GRAY), 1 , 255, cv2.THRESH_BINARY_INV)[1]
        mask_rgb = cv2.bitwise_and(mask_rgb, mask_rgb, mask=path_img_mask)
        overlay_rgb = cv2.add(mask_rgb, path_img)
    else:
        overlay_rgb = mask_rgb
    canvas[0:h, 0:w, :] = overlay_rgb

    # text
    s = 'fps:%.1f' % (userdata.current_fps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, s, (1,21), font, .5, (20,20,20),1)
    cv2.putText(canvas, s, (0,20), font, .5, (200,200,200), 1)

    # print other keys from json dict
    y = 1  # vertical offset to start at
    for key, value in path_dict.iteritems():
        y = y + 20
        # text
        if type(value) == float:
            s = '%s:%.2f' % (str(key)[0:min(len(str(key)), 10)], float(value))
        else:
            s = '%s:%s' % (str(key)[0:min(len(str(key)), 10)], str(value))
        cv2.putText(canvas, s, (w+1, y), font, .5, (20,20,20),1)
        cv2.putText(canvas, s, (w+0, y-1), font, .5, (200,200,200), 1)

    cv2.imshow('Path Detector', canvas)


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
                        help='specify the name of the path detector module with the PathDetector class (default: specified in config file; key = path_detector).')
    parser.add_argument('--config', nargs='?', default=None,
                        help='specify config file to use (default: config_server.json)')
    parser.add_argument('--display', action='store_true', default=False,
                        help='whether to visualize the computed path information (default=off)')
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
                      'path_detector': None,
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
        params['path_detector'] = args.detector # command line overrides config file
    if args.profile:
        params['profile'] = args.profile
    return params


def get_detector(params):
    # load detector
    if params['path_detector'] is None:
        logging.error('No detector module specified on command line or config file')
        sys.exit(-1)

    logging.info('Using detector module: %s' % params['path_detector'])
    mod = importlib.import_module(params['path_detector'])
    # return the module
    return mod


def setup_display(params):
    if params['display']:
        cv2.namedWindow('Path Detector')
        cv2.moveWindow('Path Detector',20 , 20)


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
        sys.exit(1)

    # Display a summary of the time used by each function
    # probably most of the time is spent waiting between acquisitions
    if parameters['profile']:
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        logging.info('Profiling data ...\n'+s.getvalue())
