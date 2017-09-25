# -*- coding: utf-8 -*-

import serial
from struct import unpack, pack

class BacarComm(serial.Serial):
    """ Implement the communication interface with the Arduino for BACAR project """
    
    def __init__(self, portName='None'):
        """ Initialise the serial port and the private variables """
        serial.Serial.__init__(self)
        self.port = portName
        self.baudrate = 115200
        self.timeout = 1
        self.parity = serial.PARITY_NONE
        self.stopbits = serial.STOPBITS_ONE
        self.bitesize = serial.EIGHTBITS
        self.x = 0
        self.y = 0
        self.u = 0
        self.v = 0

    def sendMessage(self, x,y,u,v):
        """ Send a 4-tuple message to tha arduino """
        self.write(pack('i', x))
        self.write(pack('i', y))
        self.write(pack('f', u))
        self.write(pack('f', v))

    def newMessage(self):
        """ Check if a message was received from the Arduino. """
        """ If so, reads the message and updates the private variables """
        success = False
        print(self.inWaiting())
        while (self.inWaiting() > 19) and not(success):
            if self.read(4) == "EHLO":
                x = unpack('i', self.read(4))[0]
                y = unpack('i', self.read(4))[0]
                u = unpack('f', self.read(4))[0]
                v = unpack('f', self.read(4))[0]
                success = True
        return(success)
    
    def xRead(self):
        """ return the x value from the last message received """
        return(self.x)
        
    def yRead(self):
        """ return the y value from the last message received """
        return(self.y)
        
    def uRead(self):
        """ return the u value from the last message received """
        return(self.u)

    def vRead(self):
        """ return the v value from the last message received """
        return(self.v)

