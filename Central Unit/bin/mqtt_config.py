"""
MQTT protocol definition
"""
__author__ = 'olivier'

#-----------------------------------------------------------------------------
# IMAGE SERVER

## CMD
CMD_SERVER_SET_XY = 'bacar/server_cmd/set_xy'
CMD_SERVER_GET_HSV = 'bacar/server_cmd/hsv'
CMD_SERVER_GET_RGB = 'bacar/server_cmd/rgb'
## MSG
MSG_SERVER_FPS = 'bacar/server/fps'
MSG_BOOL_ARRAY = "bacar/server/mask/boolarray"
MSG_SERVER_XY_RGB = 'bacar/server/xy/rgb'
MSG_SERVER_SIGN_ARRAY = "bacar/server/sign/bytearray" # rgb byte array
MSG_SERVER_SIGN_BBOX = "bacar/server/sign/bb/" # bounding box in the original frame [x0,y0,w,h]
MSG_SERVER_TRAFFIC_RED = 'bacar/server/traffic/red'
MSG_SERVER_TRAFFIC_ORANGE = 'bacar/server/traffic/orange'
MSG_SERVER_TRAFFIC_GREEN = 'bacar/server/traffic/green'

#-----------------------------------------------------------------------------
#  PATH DETECTOR

## CMD
CMD_SET_XY = "bacar/searchpath_cmd/set_xy"
## MSG
MSG_PATH_ARRAY = "bacar/searchpath/path/bytearray"
MSG_PATH_HEADING = "bacar/searchpath/heading/" # angle of the most probable path
MSG_PATH_DMAX = "bacar/searchpath/dmax/" # free samples along path [0,100]
MSG_PATH_JSON = "bacar/searchpath/json/" # returns path information into one single json
MSG_PATH_IMG = "bacar/searchpath/img/" # returns visual representation of path information

#-----------------------------------------------------------------------------
#  SIGN ANALYZER

## CMD

## MSG
MSG_ANALYZER_STOP_SIGN = "bacar/analyser/sign/stop" # fired when a stop sign is detected
MSG_ANALYZER_SIGN_TYPE = "bacar/analyser/sign/type" # string with the sign type
MSG_ANALYZER_SIGN_DIST = "bacar/analyser/sign/dist/" # estimated sign distance
MSG_ANALYZER_SIGN_JSON = "bacar/analyser/sign/json/" # json dict representing sign

#-----------------------------------------------------------------------------
#  DRIVER

## CMD
CMD_DRIVER_MOVE = "bacar/driver/move" # + distance in meter
CMD_DRIVER_TURN = "bacar/driver/turn" # + angle in degree
CMD_DRIVER_TURN_AND_MOVE = "bacar/driver/turn_and_move" # + json object {d: <float>, a:<float} specifying distance (in m) and angle (in degree)
CMD_DRIVER_EMERGENCY_STOP = "bacar/driver/halt"
CMD_DRIVER_SET_SPEED_AND_ANGLE = "bacar/driver/set_speed_and_angle" # + json object {s: <float>, a:<float} speed set point in m/s + heading angle set point in degrees (0 = fwd, <0 right, >0 left)

MSG_DRIVER_SEND_JSON = "bacar/driver/send"  #published by sm_server
MSG_DRIVER_STATUS_JSON = "bacar/driver/status"  #published by driver when status is received

## MSG
MSG_DRIVER_DONE = "bacar/driver/done" # last CMD is done
MSG_DRIVER_REMAINING = "bacar/driver/remaining" # + distance that remains to travel



#-----------------------------------------------------------------------------
#  REMOTE COMMAND
MSG_COMMAND = "bacar/cmd"  #consumed by sm_server to start state machine


#-----------------------------------------------------------------------------
#  STATE MACHINE

## CMD
CMD_SM = "bacar/state_machine/#"
CMD_SM_EMERGENCY_STOP = "bacar/state_machine/emergency_stop"
CMD_SM_TURN360 = "bacar/state_machine/turn360"
CMD_SM_SEQ = "bacar/state_machine/seq" # + sequence name

CMD_SET_XY = "bacar/searchpath_cmd/set_xy"


## MSG
MSG_PATH_ARRAY = "bacar/searchpath/path/bytearray"
MSG_XY_RGB = "bacar/server/xy/rgb"

#-----------------------------------------------------------------------------
#  MONITOR

## CMD

## MSG
