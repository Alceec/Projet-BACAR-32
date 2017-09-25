# -*- coding: utf-8 -*-

from bacarComm import BacarComm
from struct import unpack, pack
import time

PORT = '/dev/ttyUSB0'
PORT = 'COM3'

comm = BacarComm(PORT)
print("start")
try:
    comm.open()
    comm.sendMessage(0, 1, 3.14, 5)
    time.sleep(1)
    for a in range(3):
        if comm.newMessage():
            msg = '( ' + str(comm.xRead()) + ', ' + str(comm.yRead()) \
                + ', ' + str(comm.uRead()) + ', ' + str(comm.vRead()) +' )'
            print("Message received:" + msg)
        else:
            print("Error !")
finally:
    comm.close()
