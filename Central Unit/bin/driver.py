#!/usr/bin/python

""""
Driver bridge between state machine and the arduino-based motor driver.

Translates MQTT messages into serial port commands, and vice-versa.

--MQTT events consumed by the driver and translated to serial:

mqtt_config.MSG_DRIVER_SEND_JSON
    This is a JSON dict object with four keys: x, y, u, v where x and y
    have integer values, and u,v are floating point values. These 4 values
    are transmitted as a single tuple (x,y,u,v) to the arduino.

--MQTT events produced by the driver (translated from serial)

mqtt_config.MSG_DRIVER_STATUS_JSON
    This is a JSON dict object with four keys: x, y, u, v where x and y have
    integer values, and u, v are floating point values. This message is sent
    when the arduino communicates a four tuple (x,y,u,v) to the driver.

Note that the semantics of these four values is determined by the StateMachine
and the arduino, respectively

"""
import logging
import serial
import paho.mqtt.client as mqtt
import json
import struct
from os import sys, path
import argparse
import mqtt_config
sys.path.append(path.abspath('..'))  # add .. to search path
from tools.log_tools import create_logger
from tools.param_tools import read_param,save_param


# Constants for the arduino
ARDUINO_MSG_HEADER = "EHLO"

# Configuration of the USB port
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

MQTT_LOOPTIMEOUT = 0.008  # wait 8 ms for data to arrive on MQTT before reading from serial


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    logging.info("Connected to MQTT broker with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(mqtt_config.MSG_DRIVER_SEND_JSON, qos=0)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    # userdata contains an object of type Data
    # if we see a relevant MQTT message, we add it as an Event
    # internal queue (in userdata.messages)
    if msg.topic == mqtt_config.MSG_DRIVER_SEND_JSON:
        try:
            obj = json.loads(msg.payload.decode('utf-8'))
            x = int(obj["x"])
            y = int(obj["y"])
            u = float(obj["u"])
            v = float(obj["v"])
            logging.info("From MQTT -> arduino: x=%d,y=%d,u=%f,v=%f" % (x, y, u, v))
        except Exception as e:
            logging.error("Error decoding json payload to send to arduino. Payload = %s" % msg.payload)

        #TODO: also write sync header to avoid synchronization issues ?
        # write in little-endian format. Nano is little-endian
        data = bytearray(struct.pack('<iiff', x, y, u, v))
        ser.write(data)


def publish_serial(ser, mqtt_client):
    """Reads status from serial and publishes to MQTT. Excpects serial
       message to be an int (containing the status) and a double (containing
       the distance left to travel).
    """
    #read in little-endian format. Nano is little-endian
    fmt = '<iiff'
    numbytes = struct.calcsize(fmt)
    raw = ser.read(numbytes)
    (x, y, u, v) = struct.unpack(fmt, raw)
    payload = '{"x":%d, "y":%d, "u":%f,"v":%f}' % (x, y, u, v)
    logging.info("From arduino -> MQTT: %s" % payload)
    mqtt_client.publish(mqtt_config.MSG_DRIVER_STATUS_JSON, payload)




def setup_logging():
    log_file_name = "%s.log" % __file__
    print("Logging to " + log_file_name)
    create_logger(log_file_name, scriptname="Driver")
    # output python version
    logging.info("Running on python version" + sys.version)


def get_arguments():
    parser = argparse.ArgumentParser(description='Driver: an MQTT<->Arduino bridge')
    parser.add_argument('--config', nargs='?', default=None,
                        help='specify config file to use (default: config_server.json)')
    parser.add_argument('--usb_port', nargs='?', default=None,
                        help='specify the USB port to use to communicate to the arduino nano (default: %s)'% SERIAL_PORT)
    parser.add_argument('--usb_baudrate', nargs='?', type=int, default=None,
                        help='specify the baudrate of the USB port (default = %i)' % BAUD_RATE)
    parser.add_argument('--mqtt_host', nargs='?', default=None,
                        help='specify the hostname of the MQTT broker to connect to (default = localhost)')
    parser.add_argument('--mqtt_port', nargs='?', type=int, default=None,
                        help='specify the port of the MQTT broker to connect to (default = 1833)')
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
                      'usb_port': SERIAL_PORT,
                      'usb_baudrate': BAUD_RATE}
    params = read_param(paramsfile)
    if params is None:
        params = default_params
    else:
        # config file overrides defaults
        p = default_params.copy()
        p.update(params)
        params = p
    # command-line arguments override config file
    if args.usb_port is not None:
        params['usb_port'] = args.usb_port
    if args.usb_baudrate is not None:
        params['usb_baudrate'] = args.usb_baudrate
    if args.mqtt_host is not None:
        params['mqtt_host'] = args.mqtt_host
    if args.mqtt_port is not None:
        params['mqtt_port'] = args.mqtt_port
    return params


def get_serial(params):
    # Open serial port; fail if we are unable
    try:
        ser = serial.Serial(params['usb_port'], params['usb_baudrate'])
        if not ser.is_open:
            logging.error("Could not open serial port %s at %i bps" %
                          (params['usb_port'], params['usb_baudrate']))
            logging.error("Is the arduino connected to the orange pi ?")
            sys.exit(-1)
        return ser
    except serial.serialutil.SerialException as e:
        logging.error("Could not open serial port %s at %i bps: %s" %
                      (params['usb_port'], params['usb_baudrate'], str(e)))
        logging.error("Is the arduino connected to the orange pi ?")
        sys.exit(-1)


if __name__ == '__main__':
    setup_logging()
    args = get_arguments()

    # read parameters file
    parameters = get_parameters(args)

    # create the serial communication
    ser = get_serial(parameters)

    # create mqtt client -
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    logging.info('connecting to the broker: %s:%d'%(parameters['mqtt_host'], parameters['mqtt_port']))
    client.connect(parameters['mqtt_host'], parameters['mqtt_port'], 10)


    # required for syncin the incoming stream
    pos = 0

    while True:
        # block to process MQTT messages; continue after MQTT_LOOPTIMEOUT sec
        client.loop(timeout=MQTT_LOOPTIMEOUT)

        # check whether serial is still open
        if not ser.is_open:
            logging.error("Serial port disconnected. Exiting...")
            sys.exit(-1)

        # try and sync the incoming serial stream by recognizing HEADER;
        # then publish message
        while (ser.in_waiting > 0 and pos < len(ARDUINO_MSG_HEADER)):
            if ser.read() == ARDUINO_MSG_HEADER[pos]:
                pos = pos + 1
            else:
                pos = 0

        if pos == len(ARDUINO_MSG_HEADER) and ser.in_waiting >= 16:
            publish_serial(ser, client)
            pos = 0
