import logging
import time
from event import Event
from car import Car


# Setup up the state machine. The following code is called when the state
# machine is loaded for the first time.
logging.info('Simplistic StateMachine has been initialized')



def loop():

    event = Event.poll()
    if event is not None:

        if event.type == Event.PATH:

            tmp = event.val 
            command = tmp['command'] 
            if command == 'a' : 
                Car.send(0, 0, 2., 2.) 
            elif command == 'd' : 
                Car.send(0, 0, 2., -2.) 
            elif command == 'w' : 
                Car.send(0, 0, 2., 0.) 
            elif command == ' ' :
                Car.send(0, 0, 0., 0.)
            
        elif event.type == Event.SIGN:

            tmp = event.val
            if not tmp['sign'] == None : 
                logging.info(tmp['sign']) 
            sign = tmp['sign']
    
    
