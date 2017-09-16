#!/usr/bin/env python

"""
Send a single CMD event to the state machine via MQTT

Arguments:

    message: the message to send (string, required)

    --mqtt_host:  the hostname of the MQTT broker to connect to
                  (default: localhost)

    --mqtt_port:  the port of the MQTT broker to connect to
                  (default: 1833)

    --local:   equivalent to mqtthost=localhost, mqtt_port=1833

    --remote:  equivalent to mqtthost=bacar, mqtt_port=1833

OUTPUT MQTT events:

mqttconfig.MSG_COMMAND: command event

"""

import logging
import paho.mqtt.client as mqtt
from os import sys, path
import argparse
import mqtt_config


def setup_logging():
    log_file_name = "%s.log" % __file__
    print("Logging to " + log_file_name)
    create_logger(log_file_name)
    # output python version
    logging.info("Running on python version" + sys.version)


def get_arguments():
    parser = argparse.ArgumentParser(description='send_command: send command to state machine via MQTT')
    parser.add_argument('message', help='the command string to send to the state machine')
    parser.add_argument('--mqtt_host', nargs='?', default="localhost",
                        help='specify the hostname of the MQTT broker to connect to (default = localhost)')
    parser.add_argument('--mqtt_port', nargs='?', type=int, default=1883,
                        help='specify the port of the MQTT broker to connect to (default = 1833)')
    parser.add_argument('--local', action='store_true', default=True,
                        help='Equivalent to --mqtt_host localhost')
    parser.add_argument('--remote', action='store_true', default=False,
                        help='Equivalent to --mqtt_host bacar (MQTT broker of Orange Pi when connected to BACar Wifi network)')
    return parser.parse_args()


def get_parameters(args):
    # default parameters
    params = {'mqtt_host': 'localhost',
              'mqtt_port': 1883}
    # command-line arguments override config file
    if args.remote:
        params['mqtt_host'] = "bacar"
    if args.mqtt_host is not None:
        params['mqtt_host'] = args.mqtt_host
    if args.mqtt_port is not None:
        params['mqtt_port'] = args.mqtt_port
    params['message'] = args.message
    return params


if __name__ == '__main__':
    #    setup_logging()
    args = get_arguments()

    # read parameters file
    parameters = get_parameters(args)

    print("Sending \"%s\" to broker at %s:%s" % (parameters['message'], parameters['mqtt_host'], parameters['mqtt_port']))

    # create mqtt client -
    client = mqtt.Client()
    client.connect(parameters['mqtt_host'], parameters['mqtt_port'], 10)
    client.publish(mqtt_config.MSG_COMMAND, parameters['message'])
