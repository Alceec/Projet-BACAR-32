'''The car object allows to send a message of the form (x,y,u,v) to the
   arduino nano that is driving the car. Here,
   - x,y are integer values
   - u, v are floats
   How this is interpreted by the arduino nano depends on how you implement
   the nano
'''


class Car:

    def __init__(self):
        pass

    @staticmethod
    def send(x, y, u, v):
        # this method is is dynamically replaced by the sm_server
        pass
