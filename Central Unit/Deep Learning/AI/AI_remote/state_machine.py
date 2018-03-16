import logging
import time
from event import Event
from car import Car
import socket 

class Memory_Replay: 
    def __init__(self, Max_size) : 
        self.buffer = []
        self.max_size =Max_size
    
    def add(self, memory) : 
        
        if len(self.buffer) == self.max_size : 
            self.buffer.pop(0) 

        self.buffer.append(memory) 

        if len(self.buffer) > self.max_size :
          raise Exception('overflow')

def Action( button ) : 
  if button == 'a' : 
    Car.send(0, 0, 2., 2) 
  elif button == 'd' : 
    Car.send(0, 0,  2., -2) 
  elif button == 'w': 
    Car.send(0, 0, 2 , 0)

HOST = ''
PORT = 5555
sock = socket.socket() 
address = (HOST, PORT) 
sock.connect(address) 

memory = Memory_Replay(20) 
sign = 0
def loop():
  global memory
  global sign  
  event = Event.poll()
  if event is not None:

        if event.type == Event.PATH:
            tmp = event.val 
            
            #    LSTM
            distance = tmp['distance'] 
            distance.append(sign) 
            memory.add(distance)
            if len(memory.buffer) == memory.max_size : 
              sock.send(str(memory.buffer).encode()) 
              data = sock.recv(1024) 
              Action(data.decode())
              sign = 0 

        elif event.type == Event.SIGN:

            tmp = event.val
            if not tmp['sign'] == None : 
                logging.info(tmp['sign']) 

            if tmp['sign'] == 'LEFT' : 
              sign = -150 
            elif tmp['sign'] == 'RIGHT' : 
              sign = 150 
            elif 'STOP' in tmp['sign'] : 
              sign = 0 
            elif tmp['sign'] == None : 
              sign = 0 

