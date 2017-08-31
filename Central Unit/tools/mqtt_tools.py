"""
MQTT tools

"""

import paho
import paho.mqtt.publish as publish
import time
import numpy as np
import pickle
import json


def send_array(topic,array, hostname="localhost",port=1883):
    # send one MQTT message with the numpy array as payload
    # valid for ndim array an any type
    ndim = array.ndim
    shape = array.shape
    itemsize = array.itemsize
    bytes = bytearray(array.tostring())
    
    s = pickle.dumps({'ndim':ndim,'shape':shape,'itemsize':itemsize,
                      'bytes':bytes})
    
    publish.single(topic,bytes, hostname="localhost",port=port)

if __name__ == "__main__":

    print(paho.mqtt.__version__)
    # MQTT brocker port on localhost    
    port = 1883

    print("Sending data pickle")
    t = time.time()
    data = np.random.rand(320,240)
    send_array("test/pickle",data)       
    print('send data in %f sec'%(time.time()-t))

    print("Sending data tostring")
    t = time.time()
    data = bytearray(np.random.rand(320,240))
    publish.single("test/tostring", data, hostname="localhost",port=port)
    print('send data in %f sec'%(time.time()-t))

    for i in range(3):
        time.sleep(.1)
        print("Sending time")
        publish.single("test/time", time.time(), hostname="localhost",port=port)

    print("Sending unknown msg")
    data = bytearray(np.random.rand(100,100).tostring())
    publish.single("test/noise", data, hostname="localhost",port=port)

    #test bad format
    publish.single("test/data", "azerty", hostname="localhost",port=port)
    publish.single("test/time", "azerty", hostname="localhost",port=port)
