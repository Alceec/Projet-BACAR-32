import logging
import time
import Car_Agent
from event import Event
from car import Car
from numpy import pi
from random import randint

class InnerState : 
    IN = 1
    DONE = 0

class CarState : 
    # constants for the different states in which we can be operating
    IDLE = 1
    STOPPED = 2
    MOVING = 3 
    AUTONOMOUS = 4 
    # You can add other states here

class Direction : 
    left = -1 
    right = 1 
    forth = 1000 * 1000


# Set up the state machine. The following code is called when the state
# machine is loaded for the first time.
logging.info('State machine Initialize. Ready to rock, sir !')

#___________ Q-AGENT _____________
Schumacher = Car_Agent.BabyDriver([8, 16, 32, 16, 16], 1, 0.1, 0.009)
ReplayBuffer = Car_Agent.Replay_Memory()
total_Step = 0
#_________________________________


state = CarState.IDLE
MainAction = None 
SubMovement = None 
direction = Direction.forth 


def Rad( angle ) : 
    return angle * 2 * pi / 360


def StartTimer(interval) : 
    time_at_the_beginning = time.time()
    time_difference = ( time_at_the_beginning + interval) - time.time() 
    while ( time_difference > 0 ) : 
        yield InnerState.IN
        time_difference = ( time_at_the_beginning + interval) - time.time()
    yield InnerState.DONE

def Stop_Car() : 
    global state
    global SubMovement
    global MainAction
    Car.send(0, 0, 0, 0) 
    SubMovement = None 
    MainAction = None
    state = CarState.STOPPED 
    logging.info( "you just stopped the car" ) 



################# subfunctions ( movement ) ################################## 

def Rotate(angle) :                                   #deg                                                
    logging.info ( "the angle : "+ str( angle) ) 
    Car.send(0, 0, 0., pi/8)                          #angular speed : rad/s  ( 2*pi / 16s ) 
    timer = StartTimer(Rad(angle) / (pi / 8 )) 
    while (next(timer) == InnerState.IN) : 
        yield InnerState.IN
    Car.send(0,0, 0., 0) 
    yield InnerState.DONE

def GoForth( time_interval, speed ) :     #Speed : m/s 
    timer = StartTimer( time_interval ) 
    Car.send( Direction.forth , Direction.forth, 0, speed )
    while ( next(timer) != InnerState.DONE ) : 
        yield InnerState.IN
    yield InnerState.DONE


def SubSpiralRoutine ( speed, convergence ) : 
    for i in range ( 7200 ) : 
        Car.send(0, 0, speed, i/ convergence ) 
        logging.info( " in the spiral ! LOL ! " ) 
        yield InnerState.IN
    yield InnerState.DONE

def Turn( angle, radius, speed  ) :
    '''radius (cm)  , speed  (deg/s) , angle  (deg)'''     
    speed = Rad(speed) 
    angle = Rad(angle) 
    Car.send(radius, radius, 0, speed) 
    timer = StartTimer(angle / speed ) 
    while (next(timer) == InnerState.IN) : 
        yield InnerState.IN
    yield InnerState.DONE
    
    
    
##############################################################################

###################### Main Action ###########################################

def Square(side_length , speed ) : 
    for i in range(4) : 
        logging.info( " going forth " ) 
        yield GoForth(side_length / speed , speed)
        logging.info( " turning " )
        yield Rotate(90) 
    Stop_Car()
    yield None

def Spiral(speed, convergence_speed ) :                  #the higher the convergence speed the less time it'll take
   logging.info( "spiral being called " ) 
   yield SubSpiralRoutine(speed, convergence_speed )  
   Stop_Car()
   yield None

def ForhtMainAction (interval) : 
    yield GoForth(interval, 3) 
    Stop_Car()
    yield None 

def TurnToSide(side) :            
    yield Turn ( 90 * side , 10 , 90 / 5 ) 
    Stop_Car() 
    yield None


##################################################################################

def loop():
    
    #Constants 
    Exploration_Step = 1000
    Y = 0.99

    #Global var defined outside loop 
    global state  # define state to be a global variable
    global SubMovement
    global MainAction
    global direction 
    global Schumacher
    global ReplayBuffer
    global total_Step 


    event = Event.poll()
    if event is not None:        # only if there is some change ( instantiate action ) #me

        #+++++++++++++++++++++++++++++++++++++++++++++++++++
        if event.type == Event.CMD and event.val == "GO":
            logging.info( " Up and running, boss !" )
            state =  CarState.AUTONOMOUS
        elif event.type == Event.CMD and event.val == "SPIRAL" : 
            MainAction = Spiral(10, 10) 
            state = CarState.MOVING
        elif event.type == Event.CMD and event.val == "TEST" : 
            #state = CarState.MOVING
            #MainAction = TestAction() 
            Car.send(0, 0, 10., 30. ) 

        elif event.type == Event.CMD and event.val == "FORTH":
            MainAction = ForhtMainAction(3) 
            state = CarState.MOVING
        elif event.type == Event.CMD and event.val == "SQUARE": 
            MainAction = Square(5, 5)
            state = CarState.MOVING
        elif event.type == Event.CMD and event.val == "STOP":
            Stop_Car()

        #++++++++++++++++++++++++++++++++++++++++++++++++
        elif event.type == Event.PATH:
            dic_H = event.val
            Heading_to_take = (0, 0)
              
            if direction == Direction.forth : 
                Heading_to_take = dic_H["forth"] 

            elif direction == Direction.left : 
                Heading_to_take = dic_H["left"]

            elif direction == Direction.right : 
                Heading_to_take = dic_H["right"] 
            #######################################
            #######################################
            #######################################
            #######################################
            frame_data, Off_road = event.val
            if(total_Step < Exploration_Step ): 
                action = randint(0, 15) 
            else : 
                action = Schumacher.Move(frame_data, True)
            reward = 1 
            if Off_road : 
                reward = -10  
            if ReplayBuffer.cur_replay == [] : 
                ReplayBuffer.cur_replay.append(frame_data) 
                ReplayBuffer.cur_replay.append(action) 
                ReplayBuffer.cur_replay.append(reward) 
            else : 
                ReplayBuffer.cur_replay.append(frame_data) 
                ReplayBuffer.add(ReplayBuffer.cur_replay) 
                ReplayBuffer.cur_replay = [frame_data, action, reward]

            if total_Step >= Exploration_Step : 
                Schumacher.Train(ReplayBuffer.Sample(250), Y) 

            total_Step += 1 
            pass
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        elif event.type == Event.SIGN:
            sign = event.val['sign']
            if sign == 'right' : 
                MainAction = TurnToSide(Direction.right) 
                state = CarState.MOVING
            elif sign == 'left' : 
                MainAction = TurnToSide(Direction.left) 
                state = CarState.MOVING
            elif sign == 'stop' : 
                Stop_Car()     
            pass
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        elif event.type == Event.CAR:
            if event.val['y'] == 1 : 
                logging.info("seeing a wall") 
            pass
    else : 
        #++++++++++++++++++++++++++++++++++++++++++++++++++
        if state == CarState.MOVING : 
            if SubMovement == None or next( SubMovement ) == InnerState.DONE:        # next has to come after !!!!!!! 
                SubMovement = next(MainAction) 
                logging.info ( "Calling next subroutine " ) 
        #+++++++++++++++++++++++++++++++++++++++++++++++++++     
       
            
