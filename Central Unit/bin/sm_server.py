"""
Listen to MQTT messages sent by the image server (in particular: the boolean mask)
and emit MQTT messages that indicate path information.

Arguments:
    --machine:  the state machine module to use. This is a python module that
                has a StateMachine class with a loop() method. The default
                detector is specified in the path_server config file.

    --config: config file to use (default: server_config.json)


    --profile: whether to profile the program and display results on exit (default = off)


INPUT MQTT Events:

mqttconfig.MSG_PATH_JSON: the path dict produced by the detector routine
mqttconfig.MSG_ANALYZER_SIGN_JSON: the sign dict produced by the sign analyzer routine
mqttconfig.MSG_DRIVER_STATUS_JSON: the status dict produced by MQTT<->arduino bridge
mqttconfig.MSG_COMMAND: a command string issued by the remote computer
mqttconfig.MSG_SERVER_TRAFFIC_RED: a red traffic light has been detected by the image server
mqttconfig.MSG_SERVER_TRAFFIC_ORANGE: an orange traffic light has been detected by the image server
mqttconfig.MSG_SERVER_TRAFFIC_GREEN: a green traffic light has been detected


OUTPUT MQTT events:
mqttconfig.MSG_DRIVER_SEND_JSON: a dict to be send to the MQTT<->arduino bridge for communication to aduino.
                                 in this case, the dict must have keys x,y,u,v
                                 with x,y integer payloads and u,v, float payloads
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
from event import Event
from car import Car

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

from mqtt_config import *
from collections import deque
from types import *
import struct


class Data:
    def __init__(self, parameters):
        self.mask = None
        self.mask_ts = None # timestamp id when mask was generated
        self.mask_recv_ts = None #timestamp when mask was received
        self.fps = fps_generator(maxlen=10)
        self.current_fps = 0.0
        self.detector = None
        self.params = parameters
        self.messages = deque([])


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MSG_PATH_JSON, qos=0)
    client.subscribe(MSG_ANALYZER_SIGN_JSON, qos=0)
    client.subscribe(MSG_DRIVER_STATUS_JSON, qos=0)
    client.subscribe(MSG_COMMAND, qos=0)
    client.subscribe(MSG_SERVER_TRAFFIC_RED, qos=0)
    client.subscribe(MSG_SERVER_TRAFFIC_ORANGE, qos=0)
    client.subscribe(MSG_SERVER_TRAFFIC_GREEN, qos=0)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # userdata contains an object of type Data
    # if we see a relevant MQTT message, we add it as an Event
    # internal queue (in userdata.messages)
    if msg.topic == MSG_PATH_JSON:
        e = Event(Event.PATH, json.loads(msg.payload.decode("utf-8")))
        userdata.messages.append(e)
    elif msg.topic == MSG_ANALYZER_SIGN_JSON:
        e = Event(Event.SIGN, json.loads(msg.payload.decode("utf-8")))
        userdata.messages.append(e)
    elif msg.topic == MSG_DRIVER_STATUS_JSON:
        e = Event(Event.CAR, json.loads(msg.payload.decode("utf-8")))
        userdata.messages.append(e)
    #elif msg.topic == MSG_SERVER_TRAFFIC_RED:
    #    e = Event(Event.TRAFFICLIGHT, 'RED')
    #    userdata.messages.append(e)
    #elif msg.topic == MSG_SERVER_TRAFFIC_ORANGE:
    #    e = Event(Event.TRAFFICLIGHT, 'ORANGE')
    #    userdata.messages.append(e)
    #elif msg.topic == MSG_SERVER_TRAFFIC_GREEN:
    #    e = Event(Event.TRAFFICLIGHT, 'GREEN')
    #    userdata.messages.append(e)
    elif msg.topic == MSG_COMMAND:
        logging.info("got COMMAND from remote operator: " + msg.payload.decode("utf-8"))
        e = Event(Event.CMD, msg.payload.decode("utf-8"))
        userdata.messages.append(e)


def setup_logging():
    log_file_name = "./%s.log" % path.basename(__file__)
    scriptname = "%s" % path.basename(__file__)
    print("Logging to " + path.abspath(log_file_name))
    create_logger(log_file_name, scriptname=scriptname)
    # output python version
    logging.info("Running on python version" + sys.version.replace("\n", "\t"))


def get_arguments():
    parser = argparse.ArgumentParser(description='State Machine Server')
    parser.add_argument('--machine', nargs='?', default=None,
                        help='specify the name of the state machine module containing the StateMachine class (default: specified in config file; key=machine).')
    parser.add_argument('--config', nargs='?', default=None,
                        help='specify config file to use (default: config_server.json)')
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
                      'machine': None,
                      'profile': False}
    params = read_param(paramsfile)
    if params is None:
        params = default_params
        logging.warning("Default parameters will be used")
    else:
        # config file overrides defaults
        p = default_params.copy()
        p.update(params)
        params = p
    if args.machine is not None:
        params['machine'] = args.machine  # command line overrides config file
    if args.profile:
        params['profile'] = args.profile
    return params


def create_poll(userdata):
    # return a function that allows to poll the message queue
    def poll():
        if not userdata.messages:
            return None
        else:
            return userdata.messages.popleft()

    return poll


def create_send(mqttclient):
    '''Create a function that can be used to send a 4-tuple to the driver'''

    def send_to_driver(x, y, u, v):
        '''Asks the MQTT-to-Arduino bridge to send the four-tuple (x,y,u,v) to the
           arduino nano. Here, x,y must be integers; u,v must be floats'''
        assert type(x) is int, "x is not an integer: %r" % x
        assert type(y) is int, "y is not an integer: %r" % y
        assert type(u) is float or type(u) is int, "u is not a float: %r" % u
        assert type(v) is float or type(v) is int, "v is not a float: %r" % v
        payload = '{"x":%d, "y":%d, "u":%f,"v":%f}' % (x, y, u, v)
        mqttclient.publish(MSG_DRIVER_SEND_JSON, payload)

    return send_to_driver


def get_machine(params, userdata):
    # load machine module
    if params['machine'] is None:
        logging.error('No machine module specified on command line or config file')
        sys.exit(-1)

    logging.info('Using machine module: %s' % params['machine'])
    mod = importlib.import_module(params['machine'])
    machine = mod
    return machine


if __name__ == '__main__':
    setup_logging()
    args = get_arguments()

    # read parameters file
    parameters = get_parameters(args)

    # setup userdata object (needed in on_message handler)
    userdata = Data(parameters)

    # configure state machine to be used
    userdata.machine = get_machine(parameters, userdata)

    # ensure that a call to Event.poll() inside the state machine loop correctly
    # dequeues our message queue
    Event.poll = staticmethod(create_poll(userdata))

    # create mqtt client -
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.user_data_set(userdata)

    logging.info('connecting to the broker: %s:%d'%(parameters['mqtt_host'], parameters['mqtt_port']))
    client.connect(parameters['mqtt_host'], parameters['mqtt_port'], 10)

    # ensure that a call to Car.send() inside the state machine loop correctly
    # sends the CMD_DRIVER_SEND Mqtt message
    Car.send = staticmethod(create_send(client))

    # start profiling
    if parameters['profile']:
        logging.info('start profiling')
        pr = cProfile.Profile()
        pr.enable()

    try:
        while True:
            # block to process MQTT messages; continue after MQTT_LOOPTIMEOUT sec
            #TODO: make the timeout value a parameter ?
            MQTT_LOOPTIMEOUT = 0.005  # in seconds
            client.loop(timeout=MQTT_LOOPTIMEOUT)
            userdata.machine.msg = "Hi there"
            userdata.machine.loop()
    except KeyboardInterrupt:
        pass
    except:
        logging.exception("Exception thrown: ")
        sys.exit(1)

    # Display a summary of the time used by each function
    # probably most of the time is spent waiting between acquisitions
    if parameters['profile']:
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        logging.info('Profiling data ...\n'+s.getvalue())
