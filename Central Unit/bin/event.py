'''Event object represent events that should be received by the state machine:
 - detected paths
 - detected signs
 - status updates from the car
 - commands send by the distant operating comuter
'''


class Event:
    # constants for specifying which kind of event it is
    PATH = 0
    SIGN = 1
    CAR = 2
    CMD = 3
    TRAFFICLIGHT = 4

    def __init__(self, typ=None, val=None):
        self.type = typ
        self.val = val

    def __str__(self):
        str_typ = ""
        if (self.type == Event.PATH):
            str_typ = "PATH"
        elif (self.type == Event.SIGN):
            str_typ = "SIGN"
        elif (self.type == Event.CAR):
            str_typ = "CAR"
        elif (self.type == Event.TRAFFICLIGHT):
            str_typ = "TRAFFICLIGHT"
        else:
            str_typ = "CMD"
        return "Event(type=%s, val=%s)" % (str_typ, str(self.val))

    @staticmethod
    def poll():
        # is dynamically resolved by the sm_server
        pass
