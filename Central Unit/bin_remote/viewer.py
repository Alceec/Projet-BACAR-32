#!/usr/bin/python

"""
Listen to MQTT messages sent by the image, path, and sign servers
and visualize these messages

INPUT MQTT Events:

mqttconfig.MSG_BOOL_ARRAY: the maks array produced by the image server
mqttconfig.MSG_PATH_JSON: the JSON describing the detected path
mqttconfig.MSG_PATH_IMG: the image visualizing the detected path

Does not output any MQTT events
"""

import numpy as np
import cv2
import paho.mqtt.client as mqtt
import logging
import json
import time
import types

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from os import sys, path
sys.path.append(path.abspath('..'))

import six
from tools.param_tools import read_param,save_param
from tools.log_tools import create_logger
from tools.fps_tools import fps_generator
from tools.compression_tools import payload_to_ts_mask, payload_to_ts_ref_rgb
from tools.compression_tools import payload_to_ts_bb_signarray
from planner.planner import track_model_2
from tools.ima_tools import profile_mn

from bin.mqtt_config import *

# MQTT TOPICS THAT WE ARE SUBSCRIBING OR PUBLISHING TO
MQTT_SUBSCRIPTION_TOPICS = [MSG_BOOL_ARRAY, MSG_PATH_IMG, MSG_PATH_JSON,
                            MSG_ANALYZER_SIGN_JSON, MSG_SERVER_SIGN_ARRAY]


class Data:
    def __init__(self):
        self.mask = None
        self.mask_ts = None # timestamp id when mask was generated
        self.mask_recv_ts = None
        self.path_dict = None
        self.path_img = None
        self.path_ts = None #timestam when path was generated
        self.path_ref_ts = None # timestamp of mask that path references
        self.path_recv_ts = None
        self.sign_dict = None
        self.sign_ts = None #timestamp when sign was generated
        self.sign_ref_ts = None # timestamp of mask that sign_dict references
        self.sign_img = None
        self.sign_img_ref_ts = None # timestamp of mas that sign_img references
        self.sign_bb = None # bb of sign image
        self.fps = fps_generator(maxlen=10)
        self.current_fps = 0.0


class MQTTController:
    def __init__(self):
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.data = Data()
        self.mqtt_client.user_data_set(self.data)

    def on_connect(self, client, userdata, flags, rc):
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        for topic in MQTT_SUBSCRIPTION_TOPICS:
            self.mqtt_client.subscribe(topic, qos = 0)

    def on_message(self, client, userdata, msg):
        """Process incomming message. """
        if msg.topic == MSG_BOOL_ARRAY:
            userdata.current_fps = six.next(userdata.fps)
            userdata.mask_recv_ts = time.time()
            (userdata.mask_ts, userdata.mask) = payload_to_ts_mask(msg.payload)
            userdata.mask = userdata.mask * 255
            self.display(userdata)
        elif msg.topic == MSG_PATH_JSON:
            obj = json.loads(msg.payload)
            userdata.path_dict = obj
            userdata.path_ts = float(obj['ts'])
            userdata.path_ref_ts = float(obj['mask_ts'])
            userdata.path_recv_ts = time.time()
            self.display(userdata)
        elif msg.topic == MSG_PATH_IMG:
            ts, ref, rgb = payload_to_ts_ref_rgb(msg.payload)
            userdata.path_img = rgb
            userdata.path_img_ref = ref
            self.display(userdata)
        elif msg.topic == MSG_ANALYZER_SIGN_JSON:
            obj = json.loads(msg.payload)
            userdata.sign_dict = obj
            userdata.sign_ts = float(obj['ts'])
            userdata.sign_ref_ts = float(obj['ref_ts'])
            userdata.sign_recv_ts = time.time()
            self.display(userdata)
        elif msg.topic == MSG_SERVER_SIGN_ARRAY:
            (ts, bb, sign_img) = payload_to_ts_bb_signarray(msg.payload)
            userdata.sign_img_ref_ts = ts
            userdata.sign_bb = bb
            userdata.sign_img = sign_img
            self.display(userdata)

    def display(self, data):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # overlay path
        if data.mask is None:
            return

        (h, w) = data.mask.shape
        canvas = np.zeros((h, w*3, 3), np.uint8)
        mask_rgb = cv2.cvtColor(data.mask, cv2.COLOR_GRAY2BGR)

        if data.path_img is not None and data.path_img_ref - data.mask_ts < 0.02:
            # we only draw if the path image is not more than 20 ms older than
            # the mask image (otherwise it is too old)
            path_img_mask = cv2.threshold(cv2.cvtColor(data.path_img, cv2.COLOR_BGR2GRAY), 1 , 255, cv2.THRESH_BINARY_INV)[1]
            mask_rgb = cv2.bitwise_and(mask_rgb, mask_rgb, mask=path_img_mask)
            mask_rgb = cv2.add(mask_rgb, data.path_img)            

        if data.sign_img is not None and data.sign_img_ref_ts - data.mask_ts < 0.02:
            mask_rgb[-data.sign_bb[3]:, w-data.sign_bb[2]:w , :] = data.sign_img

        canvas[0:h, 0:w, :] = mask_rgb

        # text
        s = 'fps:%.1f' % (data.current_fps)
        cv2.putText(canvas, s, (1,21), font, .5, (20,20,20),1)
        cv2.putText(canvas, s, (0,20), font, .5, (200,200,200), 1)

        if data.path_dict is not None:
            # print other keys from json dict
            y = 1  # vertical offset to start at
            for key, value in data.path_dict.iteritems():
                y = y + 20
                if type(value) == types.FloatType:
                    s = '%s:%.2f' % (str(key)[0:min(len(str(key)), 10)], float(value))
                else:
                    s = '%s:%s' % (str(key)[0:min(len(str(key)), 10)], str(value))
                cv2.putText(canvas, s, (w+1, y), font, .5, (20,20,20),1)
                cv2.putText(canvas, s, (w+0, y-1), font, .5, (200,200,200), 1)

        if data.sign_dict is not None:
            # print other keys from json dict
            y = 21  # vertical offset to start at
            for key, value in data.sign_dict.iteritems():
                y = y + 20
                if type(value) == types.FloatType:
                    s = '%s:%.2f' % (str(key)[0:min(len(str(key)), 10)], float(value))
                else:
                    s = '%s:%s' % (str(key)[0:min(len(str(key)), 10)], str(value))
                cv2.putText(canvas, s, (2*w+1, y), font, .5, (20,20,20),1)
                cv2.putText(canvas, s, (2*w+0, y-1), font, .5, (200,200,200), 1)

        cv2.imshow('viewer', canvas)

    def loop(self):
        self.mqtt_client.loop()

    def start(self, broker="localhost", port=1883, keepalive=60):
        self.mqtt_client.connect(broker, port, keepalive)
        # do not start in background thread
        #self.mqtt_client.loop_start()
        logging.info("Connected to MQTT broker at %s:%i" % (broker, port))



if __name__ == '__main__':
    #start logging
    create_logger('%s.log'%__file__)

    # python version
    logging.info(sys.version)

    # read parameters file
    parameters = read_param('viewer_config.json')
    host = parameters['mqtt_host']
    port = parameters['mqtt_port']
    logging.info('connecting to the broker: %s:%d'%(host, port))
    # Create MQTT controller and let it connect
    mqtt_client = MQTTController()
    mqtt_client.start(broker=host, port=port)

    # create display window
    cv2.namedWindow('viewer')
    cv2.moveWindow('viewer',10,10)


    while True:
        # Wait for key (needed to display image)
        k = cv2.waitKey(1)
        mqtt_client.loop()
        if k & 0xFF == ord('h'): #hide/unhide realtime image
            hide = not hide
            logging.info("hide is %s"%str(hide))
        if k & 0xFF == ord('q'): # break the loop if "q" is keyed
            break
