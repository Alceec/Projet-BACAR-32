#!/usr/bin/python

"""
Main video processing loop
--------------------------
- open video stream 1 (the one that is opencv compatible (320x240 rgb image)
- loop over acquisition and display of the current frame
- display acquisition delay
- illustrate the profiling method using cProfile

list of the GUI commands:
- 'a' display/hide A4 calibration tool
- 'b' display birds view
- 'd' display/hide sign detection
- 'o' display overlay

- 'm' cycle over different channels
- '0' display channel 0 (RGB)

- 's' save next acquisition in ./snapshot/
- 'v' start/stop video recording (output.avi)

- 'c' get the color at x,y position
- 'R' get the color at x,y position, replace current red target color
- 'G' get the color at x,y position, replace current green target color
- 'B' get the color at x,y position, replace current blue target color

- 'f' filter on/off
- 'F' freese (keep) last acquired frame
- 'x' hide/display real-time rendering (enable higher frame rate and less CPU)
- 'q' quit the program (i.e. shutdown the server)

- '1' set the upper left A4 corner to the current cross hair position
- '2' set the upper right A4 corner to the current cross hair position
- '3' set the lower left A4 corner to the cross hair position
- '4' set the lower right A4 corner to the cross hair position

- 'h/j/u/n' move crosshair cursor by 1 pixel
- 'H/J/H/N' move crosshair cursor by 10 pixels


- '5' save parameters to json file

MQTT events:

bacar/server/fps
    sends the averaged FPS (each 10th acquisition)

bacar/server/mask/boolarray
    sends the mask image (for each frame, in bird's view mode only)

bacar/server/xy/rgb_hsv
    sends the rgb and hsv at the xy position (for each frame)


MQTT commands:

bacar/server_cmd/rgb + ''
    grab the current rgb frame
    send bacar/server/rgb/bytearray (no commpression used)

bacar/server_cmd/rgb + 'jpg'
    grab the current rgb frame
    send bacar/server/rgb/jpg (jpg commpression used)

bacar/server_cmd/hsv + ''
    grab the current hsv frame

bacar/server_cmd/set_xy + 'x,y'
    set the current x,y position of the crosshair selector (frame coordinate)
    clipped to [0,size_x] [0,size_y]

"""
import six
import numpy as np
import cv2
from time import time, sleep
import cProfile, pstats
import argparse
import struct

try: # python 2 & 3 compatibility
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import paho.mqtt.client as mqtt
import logging

from os import sys, path, makedirs
# add parent directory of current script to search path
sys.path.append(path.dirname(path.abspath(__file__))+'/..')

# add work directory (i.e., the directory from which the script is called) to search path
sys.path.append(path.abspath('.'))

from tools.param_tools import read_param,save_param
from tools.ima_tools import cyclic_inRange,rot90
from tools.log_tools import create_logger
from tools.fps_tools import fps_generator
from tools.compression_tools import ts_mask_to_payload
from tools.compression_tools import ts_bb_signarray_to_payload
from mqtt_config import *

# MQTT EVENTS
# --------------------------------------------------------------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    # callback function called after connectionhas been established with the MQTT broker
    logging.info("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("$SYS/#")
    client.subscribe("bacar/server_cmd/#", qos=0)

def on_message(client, userdata, msg):
    # callback function called after a message is received from the MQTT broker
    if msg.topic[0]=='$': # skip service messages
        pass
    if msg.topic[0]==CMD_SERVER_GET_RGB:
        if msg.payload == 'jpg':
            logging.info("sending rgb jpg image")
        else:
            logging.info("sending uncompressed rgb image")

    if msg.topic[0]==CMD_SERVER_GET_HSV:
        logging.info("sending uncompressed hsv image")

    if msg.topic[0]==CMD_SERVER_SET_XY:
        logging.info("xy set to "+msg.payload)
# --------------------------------------------------------------------------------------------------------------




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image server')
    parser.add_argument('video_file', nargs='?', default=None,
                        help='replace the camera input by a video file, played in loop')
    parser.add_argument('--vflip', action='store_true',
                        help='flip the input vertically')
    parser.add_argument('--hflip', action='store_true',
                        help='flip the input horizontally')
    parser.add_argument('--bird', action='store_true',
                        help="switch to bird's view when server starts")
    parser.add_argument('--hide', action='store_true',
                        help="hide realtime display (faster)")
    parser.add_argument('--show', action='store_true',
                        help="show realtime display")
    parser.add_argument('--filter', action='store_true',
                        help="apply noise filter")

    args = parser.parse_args()

    #start logging
    create_logger('%s.log'%__file__)

    # python version
    logging.info(sys.version)

    # OpenCV version
    logging.info('opencv '+cv2.__version__)

    if cv2.__version__ < '3':
        logging.info('opencv version 2 used')
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
    else:
        logging.info('opencv version 3 used')
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # get parametres from parameter file
    def_param = {'camera':'orange',          #'orange','picam','0'
                 'video_size':(320,240),    # e.g. OSX: (640,480)
                 'subsampling':1,           # e.g. OSX: 2
                 'rotate':0,                # 0 for landscape mode
                 'flip_vert':1,             # e.g. OSX: 0
                 'flip_horiz':1,            # e.g. OSX: 0
                 'warp_size': (210, 260),   # (200,250)
                 'offset':(340,610),        # (270,620)
                 'birdsview':0,             # 1 : starts in the birdsview mode
                 'hide':0,                  # 1 : starts in the hidden mode (faster)
                 'mqtt_host':'localhost',
                 'mqtt_port':1883,
                 'A4_calib':((58,103),(225,98),(3,209),(316,195)),
                 'A4_ref' : ((0,0),(210,0),(0,297),(210,297)),
                 'green_rgb':[47, 173, 103],  # bg mask
                 'green_hsv':[40,150,100],  # road
                 'green_tol':[50,50,50],
                 'red_hsv':[167,134,78],    # red signs
                 'red_tol':[20,69,59],
                 'blue_hsv':[109,197,107],  # blue signs
                 'blue_tol':[20,80,50],
                 'roi': [50,100]}
    parameters = read_param('config_server.json')
    if parameters is None:
        parameters = def_param
        save_param(parameters,'config_server.json')
        logging.info('save default config')

    # check if there is an argument in the cmd
    logging.info('argc:'+str(sys.argv))
    if args.video_file is not None: # if a video filename is given as first argument, it is used as source
        filename = args.video_file
        logging.info("source = file:"+path.abspath(filename))
        SRC = 'file'
    elif parameters['camera']=='picam': # source is the raspberry PICAM
        logging.info("source = PICAM")
        SRC = 'picam'
    else: # source is the opencv source available (vid1 or vid0 if no vid1)
        logging.info("source = CV")
        SRC = 'cv'

    if SRC == 'picam':
        # special import when using Raspberry PICAM
        import picamera
        from picamera.array import PiRGBArray

    # start profiling
    logging.info('start profiling')
    pr = cProfile.Profile()
    pr.enable()

    # open MQTT connection
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(parameters['mqtt_host'], parameters['mqtt_port'], 10)


    if SRC == 'file':
        cap = cv2.VideoCapture(filename)
        ret, grabbed = cap.read()
        if ret is False:
            logging.error('bad video file:'+filename)
            raise ValueError
        logging.info('video source %s is %s'%(filename,grabbed.shape))

    elif SRC == 'picam':
        # starting video capture
        cap = picamera.PiCamera()
        cap.resolution = parameters['video_size']
        cap.vflip = True
        cap.hflip = True
        #cap.framerate = 30
        rawCapture = PiRGBArray(cap, size=parameters['video_size'])


        # Let the camera warm up for 2 seconds to fix exposure speed and white balance
        logging.info("Warming up raspicam ...")
        sleep(2)
        # Now fix the values
        cap.shutter_speed = cap.exposure_speed
        cap.exposure_mode = 'off'
        g = cap.awb_gains
        cap.awb_mode = 'off'
        cap.awb_gains = g
        logging.info("Warmup finished, raspicam configured")
    else:
        # ORANGE PI camera (port1) or other opencv camera (port0)
        # starting video capture
        logging.info('try opening video camera 1 on OrangePi')
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            logging.warning('FAIL opening video camera 1 on OrangePi')
            logging.info('try opening video camera 0 instead')
            cap = cv2.VideoCapture(0)

    video_size = parameters['video_size']
    size_x,size_y = parameters['video_size']
    flip_vert = parameters['flip_vert']
    flip_horiz = parameters['flip_horiz']
    if args.vflip:
        logging.info('argument: vertical flip')
        flip_vert = not flip_vert
    if args.hflip:
        logging.info('argument: horizontal flip')
        flip_horiz = not flip_horiz
    sub = parameters['subsampling']
    image_rotate = parameters['rotate'] # set True if camera wider direction is placed vertically

    # perspective parameters
    WARP_IMAGE_SIZE = tuple(parameters['warp_size'])
    offset = np.array(parameters['offset'],dtype=np.float32)

    # total enlapsed time + total number of acquisitions
    t0 = time()
    image_count = 1

    # GUI parameters
    ov = True
    a4 = True
    detected_sign = True
    saved = False
    recording = False
    stop_recording = False
    record_max_sec = 600
    filter_rgb = False
    channel = 0
    saved_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    niter = 2
    birdsview = parameters['birdsview']
    hide = parameters['hide']
    keep = False

    sample = ''


    # overrides parameters via CMD arguments
    if args.bird:
        birdsview = True
    if args.hide:
        hide = True
    if args.show:
        hide = False
    if args.filter:
        filter_rgb = True
    if hide:
        logging.warning("HIDE mode is activated ('h' to unhide)")


    def update_perspective(src):
        # src: coordinate of the A4 sheet inside the acquired image
        # (upper left, upper right, lower left, lower right)
        # returns transform matrix

        dst = .25*(np.array(parameters['A4_ref'],dtype=np.float32)+offset)
        return cv2.getPerspectiveTransform(src,dst)

    def tol_to_range(hsv,tol):

        if hsv[0] < tol[0]:
            hmin = 179 + hsv[0]-tol[0]
            hmax = hsv[0]+tol[0]
        elif (hsv[0]+tol[0]>179):
            hmin = hsv[0]-tol[0]
            hmax = hsv[0]+tol[0]-179
        else:
            hmin = hsv[0]-tol[0]
            hmax = hsv[0]+tol[0]

        lower = np.zeros_like(hsv)
        upper = np.zeros_like(hsv)

        lower[0] = hmin
        lower[1] = max(hsv[1]-tol[1],0)
        lower[2] = max(hsv[2]-tol[2],0)

        upper[0] = hmax
        upper[1] = min(hsv[1]+tol[1],255)
        upper[2] = min(hsv[2]+tol[2],255)

        return (lower,upper)

    def callback_hue(tr):
        global parameters
        p_hsv = None

        if channels[channel][0] == 'mask+rgb':
            p_hsv = 'green_hsv'
            p_tol = 'green_tol'

        elif channels[channel][0] == 'red+rgb':
            p_hsv = 'red_hsv'
            p_tol = 'red_tol'
        elif channels[channel][0] == 'blue+rgb':
            p_hsv = 'blue_hsv'
            p_tol = 'blue_tol'

        if p_hsv:
            parameters[p_hsv] = (cv2.getTrackbarPos('hue', 'parameters'),
                                       cv2.getTrackbarPos('sat', 'parameters'),
                                       cv2.getTrackbarPos('val', 'parameters'))
            parameters[p_tol] = (cv2.getTrackbarPos('hue_tol', 'parameters'),
                                       cv2.getTrackbarPos('sat_tol', 'parameters'),
                                       cv2.getTrackbarPos('val_tol', 'parameters'))

    def callback_roi(tr):
        global roi,niter
        roi[0] = cv2.getTrackbarPos('roi_inf', 'parameters')
        roi[1] = cv2.getTrackbarPos('roi_sup', 'parameters')
        roi[2] = cv2.getTrackbarPos('x', 'parameters')
        roi[3] = cv2.getTrackbarPos('y', 'parameters')

    # process perspective transform
    src = np.array(parameters['A4_calib'],dtype=np.float32)
    M = update_perspective(src)

    # create windows and parameter sliders
    cv2.namedWindow('parameters')

    cv2.createTrackbar('x', 'parameters', 0, size_x-1, callback_roi)
    cv2.createTrackbar('y', 'parameters', 0, size_y-1, callback_roi)
    cv2.createTrackbar('roi_inf', 'parameters', parameters['roi'][0], size_y*2, callback_roi)
    cv2.createTrackbar('roi_sup', 'parameters', parameters['roi'][1], size_y*2, callback_roi)

    def set_hsv_trackbar(hsv,tol):
        cv2.createTrackbar('hue', 'parameters', hsv[0], 179, callback_hue)
        cv2.createTrackbar('hue_tol', 'parameters', tol[0], 100, callback_hue)
        cv2.createTrackbar('sat', 'parameters', hsv[1], 255, callback_hue)
        cv2.createTrackbar('sat_tol', 'parameters', tol[1], 100, callback_hue)
        cv2.createTrackbar('val', 'parameters', hsv[2], 255, callback_hue)
        cv2.createTrackbar('val_tol', 'parameters', tol[2], 100, callback_hue)

    set_hsv_trackbar(parameters['green_hsv'],parameters['green_tol'])

    roi = np.array([0,0,0,0],dtype=int)
    callback_roi('')

    cv2.namedWindow('frame')
    cv2.moveWindow('parameters', 10, 10)
    cv2.moveWindow('frame', 400, 10)


    #structuring element
    selem = np.ones((15,1),dtype=np.uint8)
    selem2 = np.ones((5,5),dtype=np.uint8)


    # main loop -------------------------------------------------------------------
    fps = fps_generator(maxlen=10) # init FPS generator
    current_fps = six.next(fps) # six is for python2 and python3 compatibility

    if SRC == 'picam':
        generator = cap.capture_continuous(rawCapture, format="bgr", use_video_port=True)
    else:
        def dumb_generator():
            while True:
                yield
        generator = dumb_generator()

    for frame in generator:
        current_fps = six.next(fps)
        image_count += 1

        if SRC == 'file': # loop
            #If the last frame is reached, reset the capture and the frame_counter
            if image_count == cap.get(CAP_PROP_FRAME_COUNT):

                image_count = 0 #Or whatever as long as it is the same as next line
                cap.set(CAP_PROP_POS_FRAMES, 0)

        # send data to MQTT
        if image_count%10 ==0:
            client.publish(MSG_SERVER_FPS, current_fps)

        if SRC == 'picam':
            # Capture BGR FRAME
            # ret, grabbed = cap.read()
            grabbed = frame.array
            rawCapture.truncate(0)
        else:
            # Capture BGR FRAME
            ret, grabbed = cap.read()
            if SRC=='file':
                sleep(0.05) #read at 20fps

        # manage image freezing
        if keep:
            grabbed = last_grabbed
        else:
            last_grabbed = grabbed.copy()


        # rotate (vertical acquisition

        if image_rotate:
            grabbed = rot90(grabbed,1)

        # flip and resize current frame
        if flip_vert:
            if flip_horiz:
                grabbed = grabbed[-1::-sub,-1::-sub,:].copy()
            else:
                grabbed = grabbed[-1::-sub,::sub,:].copy()
        else:
            if flip_horiz:
                grabbed = grabbed[::sub,::sub,:].copy()
            else:
                grabbed = grabbed[::sub,-1::-sub,:].copy()
        if birdsview:
            rgb = cv2.warpPerspective(grabbed, M, WARP_IMAGE_SIZE,borderValue = parameters['green_rgb'])
        else:
            rgb = grabbed

        # Our operations on the frame come here
        if filter_rgb:
            #rgb = cv2.bilateralFilter(rgb, 5, 20, 7)
            #rgb = cv2.medianBlur(rgb, 7)
            rgb = cv2.GaussianBlur(rgb,(5,5),0)


        # HSV color conversion
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        if birdsview:
            hsv_grabbed = cv2.cvtColor(grabbed, cv2.COLOR_BGR2HSV)
        else:
            hsv_grabbed = hsv

        # compute hsv range
        red_l,red_u = tol_to_range(parameters['red_hsv'],parameters['red_tol'])
        green_l,green_u = tol_to_range(parameters['green_hsv'],parameters['green_tol'])
        blue_l,blue_u = tol_to_range(parameters['blue_hsv'],parameters['blue_tol'])

        # Threshold the HSV image to get only green lines
        mask = cyclic_inRange(hsv, green_l, green_u)
        masked_rgb = cv2.bitwise_and(rgb, rgb, mask= mask)

        # main preprocessing is done; deterimen current timestamp - this will
        # be used as the id of all MQTT messages generated for this camera frame
        ts = time()

        # detect signs
        mask_red_sign = cyclic_inRange(hsv_grabbed, red_l, red_u)
        mask_red_sign = cv2.erode(cv2.dilate(mask_red_sign,selem),selem)

        masked_red_rgb = cv2.bitwise_and(grabbed, grabbed, mask= mask_red_sign)

        mask_red_sign[130:,:] = 0 # ignoring pixels near the car

        red_contours = cv2.findContours(mask_red_sign.copy(),
                                         mode=cv2.RETR_EXTERNAL,
                                         method=cv2.CHAIN_APPROX_SIMPLE
                                         )[-2]

        if red_contours:
            red_areas = [cv2.contourArea(ctr) for ctr in red_contours]
            max_red_area = max(red_areas)
            if max_red_area > 150:
                red_idx = red_areas.index(max_red_area)
                bb_red = cv2.boundingRect(red_contours[red_idx])
                crop_red = grabbed[bb_red[1]:bb_red[1]+bb_red[3],bb_red[0]:bb_red[0]+bb_red[2]].copy()
                if bb_red[2]>60 or bb_red[3]>60 or \
                    bb_red[2]<6 or bb_red[3]<6:
                    red_contours = None
                else:
                    # add bbox data in the first array line
                    payload = ts_bb_signarray_to_payload(ts, bb_red, crop_red)
                    client.publish(MSG_SERVER_SIGN_ARRAY, payload)
            else:
                red_contours = None

        mask_blue_sign = cyclic_inRange(hsv_grabbed, blue_l, blue_u)
        mask_blue_sign = cv2.erode(cv2.dilate(mask_blue_sign,selem),selem)

        masked_blue_rgb = cv2.bitwise_and(grabbed, grabbed, mask= mask_blue_sign)

        mask_blue_sign[130:,:] = 0 # ignoring pixels near the car

        blue_contours = cv2.findContours(mask_blue_sign.copy(),
                                         mode=cv2.RETR_EXTERNAL,
                                         method=cv2.CHAIN_APPROX_SIMPLE
                                         )[-2]
        if blue_contours:
            blue_areas = [cv2.contourArea(ctr) for ctr in blue_contours]
            max_blue_area = max(blue_areas)
            if max_blue_area > 200:
                blue_idx = blue_areas.index(max_blue_area)
                bb_blue = cv2.boundingRect(blue_contours[blue_idx])
                crop_blue = grabbed[bb_blue[1]:bb_blue[1]+bb_blue[3],bb_blue[0]:bb_blue[0]+bb_blue[2]].copy()
                if bb_blue[2]>60 or bb_blue[3]>60 or \
                   bb_blue[2]<6 or bb_blue[3]<6:
                    blue_contours = None
                else:
                    payload = ts_bb_signarray_to_payload(ts, bb_blue, crop_blue)
                    client.publish(MSG_SERVER_SIGN_ARRAY, payload)

            else:
                blue_contours = None
        #traffic light detection
        mask_lights = (np.bitwise_and(grabbed[:,:,0]>220,
                                  grabbed[:,:,1]>220)).astype(np.uint8)
        mask_lights = cv2.dilate(mask_lights,selem2)
        light_contours = cv2.findContours(mask_lights.copy(),
                                         mode=cv2.RETR_EXTERNAL,
                                         method=cv2.CHAIN_APPROX_SIMPLE
                                         )[-2]

        if light_contours:
            light_areas = [cv2.contourArea(ctr) for ctr in light_contours]
            max_light_area = max(light_areas)
            if 10 < max_light_area < 100:
                light_idx = light_areas.index(max_light_area)
                bb_light = cv2.boundingRect(light_contours[light_idx])
                crop_light = grabbed[bb_light[1]:bb_light[1]+bb_light[3],bb_light[0]:bb_light[0]+bb_light[2]].copy()
                rgb_triplet = crop_light.reshape((crop_light.shape[0]*crop_light.shape[1],3))
                med_rgb = np.mean(rgb_triplet,axis=0)

                rgb_target = np.array([[140,114,240],[170,200,206],[105,130,70]])
                d = np.linalg.norm(rgb_target - med_rgb,axis = 1)

                traffic_list = [('red',MSG_SERVER_TRAFFIC_RED),
                        ('orange',MSG_SERVER_TRAFFIC_ORANGE),
                        ('green',MSG_SERVER_TRAFFIC_GREEN)]

                light_no = np.argmin(d)
                client.publish('test',traffic_list[light_no][1])

                logging.info('traffic light detected [%s]'%traffic_list[light_no][0])

            else:
                light_contours = None

        channels = [['rgb', rgb], ['mask', mask], ['mask+rgb', masked_rgb], ['hsv', hsv],
                    ['red_s',mask_red_sign],['red+rgb',masked_red_rgb],
                    ['blue_s',mask_blue_sign],['blue+rgb',masked_blue_rgb],
                    ['hue',hsv[:,:,0]], ['saturation',hsv[:,:,1]], ['value',hsv[:,:,2]],
                    ['red', rgb[:, :, 2]], ['green', rgb[:, :, 1]], ['blue', rgb[:, :, 0]],
                    ['lights',mask_lights*255]]

        bg_image = channels[channel][1]

        # send mask to MQTT (bird's view)
        if birdsview:
            # add frustum lines
            tmp = (mask>0)
            payload = ts_mask_to_payload(ts, tmp)
            client.publish(MSG_BOOL_ARRAY, payload)

        # track road
        if birdsview:
            m,n = mask.shape
            m0 = roi[1] #sup
            m1 = roi[0] #inf

        # get XY color (rgb + hsv) regardless of the displayed image
        b,g,r =  grabbed[roi[3],roi[2],:].copy()
        # send data to MQTT broker
        payload = b'{"r":%d,"g":%d,"b":%d}'%(r,g,b)
        client.publish(MSG_SERVER_XY_RGB,payload)

        # add overlay
        bitmap = np.ascontiguousarray(bg_image)
        if ov:
            # ROI
            # cv2.rectangle(bitmap,(0,int(roi[0])),
            #                        (int(size_x),int(roi[1])),(50,50,200),1)

            # sample the current displayed image
            if not birdsview:
                # crosshair
                x = roi[2]
                y = roi[3]
                sample_hsv = hsv_grabbed[y,x].copy()
                if bitmap.ndim==2:
                    sample = bitmap[y,x].copy()
                else:
                    sample = bitmap[y,x,:].copy()

                cv2.line(bitmap, (0,y), (size_x-1,y), (255,255,255))
                cv2.line(bitmap, (x,0), (x,size_y-1), (255,255,255))

                # A4 line
                if a4:
                    cv2.polylines(bitmap, [np.int32(src[[0,1,3,2,0],:])], 0, (20,20,255))


                if detected_sign:
                    if red_contours:
                        cv2.drawContours(bitmap, red_contours, red_idx, (0, 0, 255), 3)
                        cv2.rectangle(bitmap,(bb_red[0],bb_red[1]),(bb_red[0]+bb_red[2],bb_red[1]+bb_red[3]),(0,0,255))

                    if blue_contours:
                        cv2.drawContours(bitmap, blue_contours, blue_idx, (255, 0, 0), 3)
                        cv2.rectangle(bitmap,(bb_blue[0],bb_blue[1]),(bb_blue[0]+bb_blue[2],bb_blue[1]+bb_blue[3]),(255,0,0))

                    if light_contours:
                        #cv2.drawContours(bitmap, blue_contours, blue_idx, (255, 0, 0), 3)
                        cv2.rectangle(bitmap,(bb_light[0],bb_light[1]),(bb_light[0]+bb_light[2],bb_light[1]+bb_light[3]),(0,180,180))

            else:
                sample = None
                x,y = (0,0)

            # text
            s = 'fps:%.1f-%s-%s (%d,%d)'%(current_fps, channels[channel][0],str(sample),x,y)
            cv2.putText(bitmap,s,(1,21), font, .5,(20,20,20),1)
            cv2.putText(bitmap,s,(0,20), font, .5,(200,200,200),1)

            if detected_sign:
                if blue_contours and bitmap.ndim==3 :
                    bitmap[-bb_blue[3]:,-bb_blue[2]:,:] = crop_blue

                if red_contours and bitmap.ndim==3 :
                    bitmap[-bb_red[3]:,0:bb_red[2],:] = crop_red



        if not saved:
            fname = path.abspath('./snapshot/sample%04d.png'%saved_count)
            dirname = path.dirname(fname)
            if not path.exists(dirname):
                makedirs(dirname)
            cv2.imwrite(fname,bitmap)
            logging.info('image saved [%s]'%fname)
            saved = True
            saved_count += 1

        if recording:
            if  time()-t_rec_start > record_max_sec:
                stop_recording = True
            else:
                if video_size[-1::-1] == bitmap.shape[:2]:
                    # ensure that image size has not changed since start rec
                    if bitmap.ndim==3:
                        video_writer.write(bitmap)
                    else:
                        #color_imgcallback_hue('') = cv2.cvtColor(bitmap, cv2.cv.CV_GRAY2RGB)
                        color_img = cv2.cvtColor(bitmap, cv2.COLOR_GRAY2RGB)
                        video_writer.write(color_img)

        # Display the resulting frame
        if not hide:
            cv2.imshow('frame',bitmap)

        # Wait for key
        k = cv2.waitKey(1) & 0xEFFFFF
        if k & 0xFF == ord('h') : #LEFT
            roi[2] = max(roi[2]-1,0)
        elif k & 0xFF == ord('H'): #SHIFT+LEFT
            roi[2] = max(roi[2]-10,0)
        elif k & 0xFF == ord('j'): #RIGHT
            roi[2] = min(roi[2]+1,size_x-1)
        elif k & 0xFF == ord('J'): #SHIFT+RIGHT
            roi[2] = min(roi[2]+10,size_x-10)
        elif k & 0xFF == ord('u'): #UP
            roi[3] = max(roi[3]-1,0)
        elif k & 0xFF == ord('U'): #SHIFT+UP
            roi[3] = max(roi[3]-10,0)
        elif k & 0xFF == ord('n'): #DOWN
            roi[3] = min(roi[3]+1,size_y-1)
        elif k & 0xFF == ord('N'): #SHIFT+DOWN
            roi[3] = min(roi[3]+10,size_y-1)
        elif ord('1') <= (k & 0xFF) <= ord('4') and not birdsview: #set calibration corners
            p = (k & 0xFF) - ord('1')
            x = roi[2]
            y = roi[3]
            src[p,:] = [x,y]
            M = update_perspective(src)
            logging.info('set A4 calibration:%d'%p)

        elif k & 0xFF == ord('b'): #activate bird's view
            birdsview = not birdsview
            logging.info("bird's view is %s"%str(birdsview))

        elif k & 0xFF == ord('F'): # keep last frame
            keep = not keep
            logging.info("keep last frame is %s"%str(birdsview))

        elif k & 0xFF == ord('c'): #get color at (x,y)
            logging.info("color sample: %s"%str(sample))
            if channels[channel][0] == 'hsv' and not birdsview:
                cv2.setTrackbarPos('hue', 'parameters', sample[0])
                cv2.setTrackbarPos('sat', 'parameters', sample[1])
                cv2.setTrackbarPos('val', 'parameters', sample[2])
                logging.info("target is set to: %s"%sample)

        elif k & 0xFF == ord('R'): #get color at (x,y)
            if channels[channel][0] == 'rgb' and not birdsview:
                parameters['red_hsv'] = sample_hsv
                set_hsv_trackbar(parameters['red_hsv'],parameters['red_tol'])
                channel = 5
                callback_hue('')
                logging.info("red target is set to: %s"%sample_hsv)

        elif k & 0xFF == ord('G'): #get color at (x,y)
            if channels[channel][0] == 'rgb' and not birdsview:
                parameters['green_hsv'] = sample_hsv
                set_hsv_trackbar(parameters['green_hsv'],parameters['green_tol'])
                channel = 2
                callback_hue('')
                logging.info("green target is set to: %s"%sample_hsv)

        elif k & 0xFF == ord('B'): #get color at (x,y)
            if channels[channel][0] == 'rgb' and not birdsview:
                parameters['blue_hsv'] = sample_hsv
                set_hsv_trackbar(parameters['blue_hsv'],parameters['blue_tol'])
                channel = 7
                callback_hue('')
                logging.info("blue target is set to: %s"%sample_hsv)

        elif k & 0xFF == ord('f'): #activate rgb filter
            filter_rgb = not filter_rgb
            logging.info("filter rgb is %s"%str(filter_rgb))

        elif k & 0xFF == ord('x'): #hide/unhide realtime image
            hide = not hide
            logging.info("hide is %s"%str(hide))

        elif k & 0xFF == ord('o'): #display a rectangle
            ov = not ov
            logging.info('overlay is %s'%str(ov))

        elif k & 0xFF == ord('a'): #display a rectangle
            a4 = not a4
            logging.info('A4 overlay is %s'%str(a4))

        elif k & 0xFF == ord('d'): #display detected signs
            detected_sign = not detected_sign
            logging.info('Signs overlay is %s'%str(detected_sign))

        elif k & 0xFF == ord('m'): #display mask
            channel = (channel+1)%len(channels)
            logging.info('display channel %d [%s]'%(channel,channels[channel][0]))
            # update sliders
            if channels[channel][0] == 'red+rgb':
                 set_hsv_trackbar(parameters['red_hsv'],parameters['red_tol'])
            elif channels[channel][0] == 'blue+rgb':
                set_hsv_trackbar(parameters['blue_hsv'],parameters['blue_tol'])
            elif channels[channel][0] == 'mask+rgb':
                set_hsv_trackbar(parameters['green_hsv'],parameters['green_tol'])


        elif k & 0xFF == ord('0'): #display rgb
            channel = 0
            logging.info('display channel %d [%s]'%(channel,channels[channel][0]))

        elif k & 0xFF == ord('5'): #save config

            parameters['A4_calib'] = src.tolist()
            parameters['roi'] = roi.tolist()
            parameters['target_hsv'] = [cv2.getTrackbarPos('hue', 'parameters'),
                                        cv2.getTrackbarPos('sat', 'parameters'),
                                        cv2.getTrackbarPos('val', 'parameters')]
            parameters['target_tol'] = [cv2.getTrackbarPos('hue_tol', 'parameters'),
                                        cv2.getTrackbarPos('sat_tol', 'parameters'),
                                        cv2.getTrackbarPos('val_tol', 'parameters')]
            save_param(parameters,'config_server.json')
            logging.info('save config')

        elif k & 0xFF == ord('v') or stop_recording: #record video sequence
            if recording or stop_recording:
                #stop recording
                recording = False
                stop_recording = False
                enlapsed = time()-t_rec_start
                logging.info('stop recording (total rec. time: %f sec)'%enlapsed)
                video_writer.release()

            else:
                #start recording
                video_size = bitmap.shape[:2][-1::-1]
                logging.info('start recording (max rec. time :%f sec) %s'%(record_max_sec,str(video_size)))
                # video recorder

                video_writer = cv2.VideoWriter("output.avi", fourcc, 20, video_size)

                t_rec_start = time()
                recording = True
                stop_recording = False

        elif k & 0xFF == ord('s'): #save next image
            saved = False

        elif k & 0xFF == ord('q'): # break the loop if "q" is keyed
            break

    # main loop -------------------------------------------------------------------

    avg_time = ((time()-t0)/image_count)

    logging.info('last FPS %f'%current_fps)
    logging.info('average time: %f [s/loop]'%avg_time)

    if cap.isOpened():
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    else:
        logging.warning('no camera found')

    # Display a summary of the time used by each function
    # probably most of the time is spent waiting between acquisitions
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logging.info('Profiling data ...\n'+s.getvalue())
