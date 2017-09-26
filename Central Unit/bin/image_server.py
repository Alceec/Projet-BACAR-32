#!/usr/bin/python

"""
Main video processing loop
--------------------------
- open video stream 1 (the one that is opencv compatible (320x240 rgb image)
- open video file and play it in loop
- loop over acquisition and display of the current frame

List of the GUI keyboard commands:
----------------------------------

- 'a' display/hide A4 calibration tool
- 'b' display birds view
- 'd' display/hide sign detection
- 'o' display overlay

- 'm' cycle over different channels
- '0' display channel 0 (RGB)

- 's' save next acquisition in ./snapshot/
- 'v' start/stop video recording (output.avi)

- 'h/j/u/n' move crosshair cursor by 1 pixel
- 'H/J/H/N' move crosshair cursor by 10 pixels

- 'c' get the color at x,y position
- 'R' get the color at x,y position, replace current red target color
- 'G' get the color at x,y positionf, replace current green target color
- 'B' get the color at x,y position, replace current blue target color

- 'f' filter on/off
- 'F' freeze/resume last acquired frame
- 'r' pause/resume real-time rendering (enable higher frame rate and less CPU)

- '1' set the upper left A4 corner to the current cross hair position
- '2' set the upper right A4 corner to the current cross hair position
- '3' set the lower left A4 corner to the cross hair position
- '4' set the lower right A4 corner to the cross hair position

- '5' save parameters to json file

- 'q' quit the program (i.e. shutdown the server)

"""

import six
import numpy as np
import cv2
from time import time, sleep
import cProfile
import pstats
import argparse

try:  # python 2 & 3 compatibility
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import paho.mqtt.client as mqtt
import logging

from os import sys, path, makedirs
# add parent directory of current script to search path
sys.path.append(path.dirname(path.abspath(__file__)) + '/..')

# add work directory (i.e., the directory from which the script is called) to search path
sys.path.append(path.abspath('.'))

from tools.param_tools import read_param, save_param
from tools.ima_tools import cyclic_inRange, rot90
from tools.log_tools import create_logger
from tools.fps_tools import fps_generator
from tools.compression_tools import ts_mask_to_payload
from tools.compression_tools import ts_bb_signarray_to_payload
from mqtt_config import *

# MQTT EVENTS
# --------------------------------------------------------------------------------------------------------------
"""
MQTT events:
------------

bacar/server/fps
    sends the averaged FPS (each 10th acquisition)

bacar/server/mask/boolarray
    sends the mask image (for each frame, in bird's view mode only)

bacar/server/xy/rgb_hsv
    sends the rgb and hsv at the xy position (for each frame)

bacar/server/traffic/red
bacar/server/traffic/green
    sends a message when a traffic light is detected

"""


def on_connect(client, userdata, flags, rc):
    # callback function called after connectionhas been established with the MQTT broker
    logging.info("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    # client.subscribe("$SYS/#")
    # client.subscribe("bacar/server_cmd/#", qos=0)


def on_message(client, userdata, msg):
    # callback function called after a message is received from the MQTT broker
    if msg.topic[0] == '$':  # skip service messages
        pass

# --------------------------------------------------------------------------------------------------------------
# MONOSTABLE

def monostable(sec):
    # generator that allows to group consecutive pulses (mask new pulse <sec)
    # if time since previous call is inferior to sec, return False
    t = time() - sec
    while True:
        d = time() - t
        t = time()
        if (d <= sec):
            yield False
        else:
            yield True

# --------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image server',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
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

    # start logging
    create_logger('%s.log' % __file__)

    # python version
    logging.info(sys.version)

    # OpenCV version
    logging.info('opencv ' + cv2.__version__)

    if cv2.__version__ < '3':
        logging.info('opencv version 2 used')
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
        fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    else:
        logging.info('opencv version 3 used')
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # get parametres from parameter file

    def_param = {'camera':'orange',          #'orange','picam','0'
                 'video_size':(320,240),    # e.g. OSX: (640,480)
                 'subsampling':1,           # e.g. OSX: 2
                 'rotate':0,                # 0 for landscape mode
                 'flip_vert':1,             # e.g. OSX: 0
                 'flip_horiz':1,            # e.g. OSX: 0
                 'warp_size': (210, 260),   # (200,250)
                 'offset': (340, 610),        # (270,620)
                 'birdsview': 0,             # 1 : starts in the birdsview mode
                 # 1 : starts in the hidden mode (console mode only > faster)
                 'hide': 0,
                 # 1 : starts in the render mode (each frame is displayed if hide ==0)
                 'render': 1,
                 'filter': 1,                # 1 : starts with the noise filter ON
                 'mqtt_host': 'localhost',
                 'mqtt_port': 1883,
                 'A4_calib': ((58, 103), (225, 98), (3, 209), (316, 195)),
                 'A4_ref': ((0, 0), (210, 0), (0, 297), (210, 297)),
                 'green_rgb': [47, 173, 103],  # bg mask
                 'green_hsv': [40, 150, 100],  # road
                 'green_tol': [50, 50, 50],
                 'red_hsv': [167, 134, 78],    # red signs
                 'red_tol': [20, 69, 59],
                 'blue_hsv': [109, 197, 107],  # blue signs
                 'blue_tol': [20, 80, 50],
                 'roi': [50, 100],
                 # arena description file (used in simulator)
                 "arena_json": "./data/test_arena9s.json"
                 }
    parameters = read_param('config_server.json')
    if parameters is None:
        parameters = def_param
        save_param(parameters, 'config_server.json')
        logging.info('save default config')

    # check if there is an argument in the cmd
    logging.info('argc:' + str(sys.argv))
    if args.video_file is not None:  # if a video filename is given as first argument, it is used as source
        filename = args.video_file
        logging.info("source = file:" + path.abspath(filename))
        SRC = 'file'
    elif parameters['camera'] == 'picam':  # source is the raspberry PICAM
        logging.info("source = PICAM")
        SRC = 'picam'
    else:  # source is the opencv source available (vid1 or vid0 if no vid1)
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
            logging.error('bad video file:' + filename)
            raise ValueError
        logging.info('video source %s is %s' % (filename, grabbed.shape))

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

    video_size = parameters.get('video_size', def_param['video_size'])
    size_x, size_y = parameters['video_size']
    flip_vert = parameters.get('flip_vert', def_param['flip_vert'])
    flip_horiz = parameters.get('flip_horiz', def_param['flip_horiz'])
    if args.vflip:
        logging.info('argument: forcing vertical flip')
        flip_vert = not flip_vert
    if args.hflip:
        logging.info('argument: forcing horizontal flip')
        flip_horiz = not flip_horiz
    sub = parameters.get('subsampling', def_param['subsampling'])
    image_rotate = parameters.get('rotate', def_param['subsampling'])

    # perspective parameters
    WARP_IMAGE_SIZE = tuple(parameters.get(
        'warp_size', def_param['warp_size']))
    offset = np.array(parameters.get(
        'offset', def_param['offset']), dtype=np.float32)

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
    channel = 0
    saved_count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    niter = 2
    # global parameters from the config file (if non existing, default values are used)
    filter_rgb = parameters.get('filter', def_param['filter'])
    birdsview = parameters.get('birdsview', def_param['birdsview'])
    hide = parameters.get('hide', def_param['hide'])
    freeze = False

    red_monostable = monostable(3)
    green_monostable = monostable(3)

    sample = ''

    # overrides parameters via CMD arguments
    if args.bird:
        birdsview = True
        logging.info("Bird's view activated by command line")
    if args.hide:
        hide = True
        logging.info("Hide view activated by command line")
    if args.show:
        hide = False
        logging.info("Hide view de-activated by command line")
    if args.filter:
        filter_rgb = True
        logging.info("Noise reduction filter activated by command line")
    if hide:
        logging.warning(
            "HIDE mode is activated (console mode only/no display nor keyboard interaction)")

    if hide:
        render = False
    else:
        render = parameters.get('render', def_param['render'])

    def update_perspective(src):
        # src: coordinate of the A4 sheet inside the acquired image
        # (upper left, upper right, lower left, lower right)
        # returns transform matrix

        dst = .25 * (np.array(parameters['A4_ref'], dtype=np.float32) + offset)
        return cv2.getPerspectiveTransform(src, dst)

    def tol_to_range(hsv, tol):

        if hsv[0] < tol[0]:
            hmin = 179 + hsv[0] - tol[0]
            hmax = hsv[0] + tol[0]
        elif (hsv[0] + tol[0] > 179):
            hmin = hsv[0] - tol[0]
            hmax = hsv[0] + tol[0] - 179
        else:
            hmin = hsv[0] - tol[0]
            hmax = hsv[0] + tol[0]

        lower = np.zeros_like(hsv)
        upper = np.zeros_like(hsv)

        lower[0] = hmin
        lower[1] = max(hsv[1] - tol[1], 0)
        lower[2] = max(hsv[2] - tol[2], 0)

        upper[0] = hmax
        upper[1] = min(hsv[1] + tol[1], 255)
        upper[2] = min(hsv[2] + tol[2], 255)

        return (lower, upper)

    channels = [['']]
    def callback_hue(tr):
        global parameters,channels
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

    def getContourStat(contour, image):
        # returns mean and stdev for the contour int the given image
        m, n = image.shape[:2]
        mask = np.zeros((m, n), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        if image.ndim == 2:
            mean, stddev = cv2.meanStdDev(image, mask=mask)
            return mean, stddev
        if image.ndim == 3:
            mean = np.zeros(3)
            stddev = np.zeros(3)
            mean[0], stddev[0] = cv2.meanStdDev(image[:, :, 0], mask=mask)
            mean[1], stddev[1] = cv2.meanStdDev(image[:, :, 1], mask=mask)
            mean[2], stddev[2] = cv2.meanStdDev(image[:, :, 2], mask=mask)
            return mean, stddev

    def rgb_not_gray(rgb):
        # return true if not gray
        s = np.sort(rgb + .1)
        if (s[2] / s[0] < 1.3):
            return False
        else:
            return True

    # process perspective transform
    src = np.array(parameters['A4_calib'], dtype=np.float32)
    perspectiveM = update_perspective(src)

    if not hide:
        # create windows and parameter sliders

        cv2.namedWindow('parameters')

        cv2.createTrackbar('hue', 'parameters', 0, 179, callback_hue)
        cv2.createTrackbar('hue_tol', 'parameters',0, 100, callback_hue)
        cv2.createTrackbar('sat', 'parameters', 0, 255, callback_hue)
        cv2.createTrackbar('sat_tol', 'parameters',0, 100, callback_hue)
        cv2.createTrackbar('val', 'parameters', 0, 255, callback_hue)
        cv2.createTrackbar('val_tol', 'parameters',0, 100, callback_hue)

        def set_hsv_trackbar(hsv, tol):
            cv2.setTrackbarPos('hue', 'parameters', hsv[0])
            cv2.setTrackbarPos('hue_tol', 'parameters',tol[0])
            cv2.setTrackbarPos('sat', 'parameters', hsv[1])
            cv2.setTrackbarPos('sat_tol', 'parameters',tol[1])
            cv2.setTrackbarPos('val', 'parameters', hsv[2])
            cv2.setTrackbarPos('val_tol', 'parameters',tol[2])

        set_hsv_trackbar(parameters['green_hsv'], parameters['green_tol'])

    roi = np.array([0, 0, 0, 0], dtype=int)

    if not hide:
        cv2.namedWindow('frame')
        cv2.moveWindow('parameters', 10, 10)
        cv2.moveWindow('frame', 400, 10)

    # structuring element
    selem = np.ones((15, 1), dtype=np.uint8)
    selem2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # main loop -------------------------------------------------------------------
    fps = fps_generator(maxlen=10)  # init FPS generator
    current_fps = six.next(fps)  # six is for python2 and python3 compatibility

    if SRC == 'picam':
        generator = cap.capture_continuous(
            rawCapture, format="bgr", use_video_port=True)
    else:
        def dumb_generator():
            while True:
                yield
        generator = dumb_generator()

    for frame in generator:
        current_fps = six.next(fps)
        image_count += 1

        if SRC == 'file':  # loop
            # If the last frame is reached, reset the capture and the frame_counter
            if image_count == cap.get(CAP_PROP_FRAME_COUNT):

                image_count = 0  # Or whatever as long as it is the same as next line
                cap.set(CAP_PROP_POS_FRAMES, 0)

        # send data to MQTT
        if image_count % 10 == 0:
            client.publish(MSG_SERVER_FPS, current_fps)

        if SRC == 'picam':
            # Capture BGR FRAME
            # ret, grabbed = cap.read()
            grabbed = frame.array
            rawCapture.truncate(0)
        else:
            # Capture BGR FRAME
            ret, grabbed = cap.read()
            if SRC == 'file':
                sleep(0.05)  # read at 20fps

        # manage image freezing
        if freeze:
            grabbed = last_grabbed
        else:
            last_grabbed = grabbed.copy()

        # rotate (vertical acquisition
        if image_rotate:
            grabbed = rot90(grabbed, 1)

        # flip and resize current frame
        if flip_vert:
            if flip_horiz:
                grabbed = grabbed[-1::-sub, -1::-sub, :].copy()
            else:
                grabbed = grabbed[-1::-sub, ::sub, :].copy()
        else:
            if flip_horiz:
                grabbed = grabbed[::sub, ::sub, :].copy()
            else:
                grabbed = grabbed[::sub, -1::-sub, :].copy()
        rgb_bv = cv2.warpPerspective(
            grabbed, perspectiveM, WARP_IMAGE_SIZE, borderValue=parameters['green_rgb'])
        rgb = grabbed

        # Our operations on the frame come here
        if filter_rgb:
            #rgb = cv2.bilateralFilter(rgb, 5, 20, 7)
            #rgb = cv2.medianBlur(rgb, 7)
            rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
            rgb_bv = cv2.GaussianBlur(rgb_bv, (5, 5), 0)

        # HSV color conversion
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        hsv_bv = cv2.cvtColor(rgb_bv, cv2.COLOR_BGR2HSV)

        # compute hsv range
        red_l, red_u = tol_to_range(
            parameters['red_hsv'], parameters['red_tol'])
        green_l, green_u = tol_to_range(
            parameters['green_hsv'], parameters['green_tol'])
        blue_l, blue_u = tol_to_range(
            parameters['blue_hsv'], parameters['blue_tol'])

        # Threshold the HSV image to get only green lines
        mask = cyclic_inRange(hsv, green_l, green_u)
        masked_rgb = cv2.bitwise_and(rgb, rgb, mask=mask)

        mask_bv = cyclic_inRange(hsv_bv, green_l, green_u)
        masked_rgb_bv = cv2.bitwise_and(rgb_bv, rgb_bv, mask=mask_bv)

        def detect_signs(rgb, hsv, lower, upper):
            # detect signs
            # returns contour list, idx of the best candidates if any
            # and the BB + cropped rgb
            selem = np.ones((15, 1), dtype=np.uint8)
            mask_sign = cyclic_inRange(hsv, lower, upper)
            mask_sign = cv2.erode(cv2.dilate(mask_sign, selem), selem)

            mask_sign[130:, :] = 0  # ignoring pixels near the car

            sign_contours = cv2.findContours(mask_sign.copy(),
                                             mode=cv2.RETR_EXTERNAL,
                                             method=cv2.CHAIN_APPROX_SIMPLE
                                             )[-2]
            bb = None
            idx = None
            crop_red = None
            if sign_contours:
                sign_areas = [cv2.contourArea(ctr) for ctr in sign_contours]
                max_sign_area = max(sign_areas)
                if max_sign_area > 150:
                    idx = sign_areas.index(max_sign_area)
                    bb = cv2.boundingRect(sign_contours[idx])
                    crop_red = rgb[bb[1]:bb[1] + bb[3],
                                   bb[0]:bb[0] + bb[2]].copy()
                    if bb[2] > 60 or bb[3] > 60 or \
                            bb[2] < 6 or bb[3] < 6:
                        sign_contours = None
                        bb = None
                        crop_red = None
                else:
                    sign_contours = None
                    bb = None
                    crop_red = None

            return (mask_sign, sign_contours, idx, bb, crop_red)

        # current Time stamp
        ts = time()

        # STOP sign detection
        (mask_red_sign, red_contours, red_idx, bb_red,
         crop_red) = detect_signs(grabbed, hsv, red_l, red_u)
        masked_red_rgb = cv2.bitwise_and(grabbed, grabbed, mask=mask_red_sign)

        if bb_red:
            payload = ts_bb_signarray_to_payload(ts, bb_red, crop_red)
            client.publish(MSG_SERVER_SIGN_ARRAY, payload)

        # LEFT/RIGHT sign detection
        (mask_blue_sign, blue_contours, blue_idx, bb_blue,
         crop_blue) = detect_signs(grabbed, hsv, blue_l, blue_u)
        masked_blue_rgb = cv2.bitwise_and(
            grabbed, grabbed, mask=mask_blue_sign)

        if bb_blue:
            payload = ts_bb_signarray_to_payload(ts, bb_blue, crop_blue)
            client.publish(MSG_SERVER_SIGN_ARRAY, payload)

        # traffic light detection
        mask_lights = (np.bitwise_and(grabbed[:, :, 0] > 200,
                                      grabbed[:, :, 1] > 220)).astype(np.uint8)
        mask_lights = cv2.dilate(mask_lights, selem2)
        light_contours = cv2.findContours(mask_lights.copy(),
                                          mode=cv2.RETR_EXTERNAL,
                                          method=cv2.CHAIN_APPROX_SIMPLE
                                          )[-2]

        # TRAFFIC LIGHTS
        if light_contours:
            scores = np.zeros((len(light_contours), 2))
            rec = np.zeros((len(light_contours), 4))
            rec[:, 0] = range(len(light_contours))
            for i, ctr in enumerate(light_contours):
                # compute contour centroid
                momentsM = cv2.moments(ctr)
                try:
                    cX = int(momentsM["m10"] / momentsM["m00"])
                    cY = int(momentsM["m01"] / momentsM["m00"])
                    # crosshair position
                    xc = roi[2]
                    yc = roi[3]
                except ZeroDivisionError:
                    pass

                if 15 < cv2.contourArea(ctr) < 200:

                    x, y, w, h = cv2.boundingRect(ctr)
                    try:
                        aspect_ratio = float(w) / h
                    except ZeroDivisionError:
                        aspect_ratio = 10.

                    # aspect ratio tolerance [7/10 --> 10/7]
                    if (.7 <= aspect_ratio <= 1.42) and (w >= 6) and (h >= 6):
                        mean_blob, _ = getContourStat(ctr, hsv)  # grabbed

                        bb_light = cv2.boundingRect(ctr)

                        traffic_list = [('red', MSG_SERVER_TRAFFIC_RED, red_monostable),
                                        ('green', MSG_SERVER_TRAFFIC_GREEN, green_monostable)]

                        # save crop
                        crop_light_rgb = grabbed[bb_light[1]:bb_light[1] +
                                                 bb_light[3], bb_light[0]:bb_light[0] + bb_light[2]].copy()
                        mask_white = np.bitwise_and(
                            crop_light_rgb[:, :, 0] > 200, crop_light_rgb[:, :, 1] > 200, crop_light_rgb[:, :, 1] > 200)

                        def is_green(ima):
                            try:
                                test = ima[:, :, 1].mean(
                                ) / ima[:, :, 2].mean() > 1.1 and ima[:, :, 0].mean() < 140
                                return test
                            except ZeroDivisionError:
                                return False

                        def is_red(ima):
                            try:
                                test = ima[:, :, 2].mean(
                                ) / ima[:, :, 0].mean() > 1.5 and ima[:, :, 0].mean() < 150
                                return test
                            except ZeroDivisionError:
                                return False

                        def green_red_ratio(ima):
                            try:
                                ratio = ima[:, :, 1].mean() / ima[:, :, 2].mean()
                                return ratio
                            except ZeroDivisionError:
                                return 0.5

                        def contrast_1d(y):
                            m = len(y)
                            v_cent = y[int(m / 2) - 1:int(m / 2) + 1].mean()
                            v_out = .5 * (y[0] + y[-1])
                            return v_cent / v_out

                        def contrast_2d(ima):
                            y0 = ima[:, :, 0].mean(axis=1)
                            y1 = ima[:, :, 0].mean(axis=1)
                            return np.min((contrast_1d(y1), contrast_1d(y0)))

                        # check if both contrast and color are satisfied
                        ctrst = contrast_2d(crop_light_rgb)
                        if ctrst > 1.8 and (is_green(crop_light_rgb) or is_red(crop_light_rgb)):
                            rec[i, 1:] = ctrst
                            scores[i, :] = (
                                ctrst, green_red_ratio(crop_light_rgb))

                            if abs(xc - cX) < 5 and abs(yc - cY) < 5:
                                print('blob score: ', scores[i, :])

            # find best candidate
            maxscore = np.max(scores[:, 0])
            best = np.argmax(scores[:, 0])

            # closest target (red/green)
            if scores[best, 1] > 1:
                best_color = 1
            else:
                best_color = 0

            bb_light = cv2.boundingRect(light_contours[best])
            if maxscore > 2:
                mono = six.next(traffic_list[best_color][2])
                if mono:
                    client.publish(traffic_list[best_color][1], '')
                    logging.info('traffic light detected [%s]' % (
                        traffic_list[best_color][0]))
            else:
                light_contours = None

        channels = [['rgb', rgb], ['mask', mask], ['mask+rgb', masked_rgb],
                    ['red+rgb', masked_red_rgb],
                    ['blue+rgb', masked_blue_rgb],
                    ['lights', mask_lights * 255]]

        channels_bv = [['rgb', rgb_bv], [
            'mask', mask_bv], ['mask+rgb', masked_rgb_bv]]

        if birdsview:
            bg_image = channels_bv[channel][1]
        else:
            bg_image = channels[channel][1]

        # send mask to MQTT (bird's view)
        tmp = (mask_bv > 0)
        payload = ts_mask_to_payload(ts, tmp)
        client.publish(MSG_BOOL_ARRAY, payload)

        # add overlay
        bitmap = np.ascontiguousarray(bg_image)
        if ov:
            # ROI

            # sample the current displayed image
            if not birdsview:
                # crosshair
                x = roi[2]
                y = roi[3]
                sample_hsv = hsv[y, x].copy()
                sample_rgb = rgb[y, x].copy()
                if bitmap.ndim == 2:
                    sample = bitmap[y, x].copy()
                else:
                    sample = bitmap[y, x, :].copy()

                cv2.line(bitmap, (0, y), (size_x - 1, y), (255, 255, 255))
                cv2.line(bitmap, (x, 0), (x, size_y - 1), (255, 255, 255))

                # A4 line
                if a4:
                    cv2.polylines(
                        bitmap, [np.int32(src[[0, 1, 3, 2, 0], :])], 0, (20, 20, 255))

                if detected_sign:
                    if red_contours:
                        cv2.drawContours(bitmap, red_contours,
                                         red_idx, (0, 0, 255), 3)
                        cv2.rectangle(
                            bitmap, (bb_red[0], bb_red[1]), (bb_red[0] + bb_red[2], bb_red[1] + bb_red[3]), (0, 0, 255))

                    if blue_contours:
                        cv2.drawContours(bitmap, blue_contours,
                                         blue_idx, (255, 0, 0), 3)
                        cv2.rectangle(bitmap, (bb_blue[0], bb_blue[1]), (
                            bb_blue[0] + bb_blue[2], bb_blue[1] + bb_blue[3]), (255, 0, 0))

                    if light_contours:
                        cv2.rectangle(bitmap, (bb_light[0], bb_light[1]), (
                            bb_light[0] + bb_light[2], bb_light[1] + bb_light[3]), (0, 180, 180))

            else:
                sample = None
                x, y = (0, 0)

            # text
            if birdsview:
                s = 'fps:%.1f-%s-%s (%d,%d)' % (current_fps,
                                                channels_bv[channel][0], str(sample), x, y)
            else:
                s = 'fps:%.1f-%s-%s (%d,%d)' % (current_fps,
                                                channels[channel][0], str(sample), x, y)

            cv2.putText(bitmap, s, (1, 21), font, .5, (20, 20, 20), 1)
            cv2.putText(bitmap, s, (0, 20), font, .5, (200, 200, 200), 1)

            if detected_sign:
                if blue_contours and bitmap.ndim == 3:
                    bitmap[-bb_blue[3]:, -bb_blue[2]:, :] = crop_blue

                if red_contours and bitmap.ndim == 3:
                    bitmap[-bb_red[3]:, 0:bb_red[2], :] = crop_red

        if not saved:
            fname = path.abspath('./snapshot/sample%04d.png' % saved_count)
            dirname = path.dirname(fname)
            if not path.exists(dirname):
                makedirs(dirname)
            cv2.imwrite(fname, bitmap)
            logging.info('image saved [%s]' % fname)
            saved = True
            saved_count += 1

        if recording:
            if time() - t_rec_start > record_max_sec:
                stop_recording = True
            else:
                if video_size[-1::-1] == bitmap.shape[:2]:
                    # ensure that image size has not changed since start rec
                    if bitmap.ndim == 3:
                        video_writer.write(bitmap)
                    else:
                        color_img = cv2.cvtColor(bitmap, cv2.COLOR_GRAY2RGB)
                        video_writer.write(color_img)

        if not hide:
            # Display the resulting frame
            if render:
                cv2.imshow('frame', bitmap)

            # Wait for key
            k = cv2.waitKey(1) & 0xEFFFFF
            if k & 0xFF == ord('h'):  # LEFT
                roi[2] = max(roi[2] - 1, 0)
            elif k & 0xFF == ord('H'):  # SHIFT+LEFT
                roi[2] = max(roi[2] - 10, 0)
            elif k & 0xFF == ord('j'):  # RIGHT
                roi[2] = min(roi[2] + 1, size_x - 1)
            elif k & 0xFF == ord('J'):  # SHIFT+RIGHT
                roi[2] = min(roi[2] + 10, size_x - 10)
            elif k & 0xFF == ord('u'):  # UP
                roi[3] = max(roi[3] - 1, 0)
            elif k & 0xFF == ord('U'):  # SHIFT+UP
                roi[3] = max(roi[3] - 10, 0)
            elif k & 0xFF == ord('n'):  # DOWN
                roi[3] = min(roi[3] + 1, size_y - 1)
            elif k & 0xFF == ord('N'):  # SHIFT+DOWN
                roi[3] = min(roi[3] + 10, size_y - 1)
            elif ord('1') <= (k & 0xFF) <= ord('4') and not birdsview:  # set calibration corners
                p = (k & 0xFF) - ord('1')
                x = roi[2]
                y = roi[3]
                src[p, :] = [x, y]
                perspectiveM = update_perspective(src)
                logging.info('set A4 calibration:%d' % p)

            elif k & 0xFF == ord('b'):  # activate bird's view
                birdsview = not birdsview
                logging.info("bird's view is %s" % str(birdsview))
                channel = 0

            elif k & 0xFF == ord('F'):  # keep last frame
                freeze = not freeze
                logging.info("keep last frame is %s" % str(birdsview))

            elif k & 0xFF == ord('c'):  # get color at (x,y)
                logging.info("color sample: %s" % str(sample))
                # if channels[channel][0] == 'hsv' and not birdsview:
                #     cv2.setTrackbarPos('hue', 'parameters', sample[0])
                #     cv2.setTrackbarPos('sat', 'parameters', sample[1])
                #     cv2.setTrackbarPos('val', 'parameters', sample[2])
                #     logging.info("target is set to: %s" % sample)

            elif k & 0xFF == ord('R'):  # get color at (x,y)
                if channels[channel][0] == 'rgb' and not birdsview:
                    parameters['red_hsv'] = sample_hsv
                    set_hsv_trackbar(
                        parameters['red_hsv'], parameters['red_tol'])
                    channel = 3
                    callback_hue('')
                    logging.info("red target is set to: %s" % sample_hsv)

            elif k & 0xFF == ord('G'):  # get color at (x,y)
                if channels[channel][0] == 'rgb' and not birdsview:
                    parameters['green_hsv'] = sample_hsv
                    parameters['green_rgb'] = sample_rgb.tolist()
                    set_hsv_trackbar(
                        parameters['green_hsv'], parameters['green_tol'])
                    channel = 2
                    callback_hue('')
                    logging.info("green target is set to: %s" % sample_hsv)

            elif k & 0xFF == ord('B'):  # get color at (x,y)
                if channels[channel][0] == 'rgb' and not birdsview:
                    parameters['blue_hsv'] = sample_hsv
                    set_hsv_trackbar(
                        parameters['blue_hsv'], parameters['blue_tol'])
                    channel = 4
                    callback_hue('')
                    logging.info("blue target is set to: %s" % sample_hsv)

            elif k & 0xFF == ord('f'):  # activate rgb filter
                filter_rgb = not filter_rgb
                logging.info("filter rgb is %s" % str(filter_rgb))

            elif k & 0xFF == ord('r'):  # pause/resume realtime image rendering
                render = not render
                logging.info("rendering is %s" % str(render))

            elif k & 0xFF == ord('o'):  # display a rectangle
                ov = not ov
                logging.info('overlay is %s' % str(ov))

            elif k & 0xFF == ord('a'):  # display a rectangle
                a4 = not a4
                logging.info('A4 overlay is %s' % str(a4))

            elif k & 0xFF == ord('d'):  # display detected signs
                detected_sign = not detected_sign
                logging.info('Signs overlay is %s' % str(detected_sign))

            elif k & 0xFF == ord('m'):  # display mask
                if birdsview:
                    channel = (channel + 1) % len(channels_bv)
                    logging.info("display channel bird's view %d [%s]" % (
                        channel, channels_bv[channel][0]))
                else:
                    channel = (channel + 1) % len(channels)
                    logging.info('display channel %d [%s]' % (
                        channel, channels[channel][0]))
                    # update sliders
                    if channels[channel][0] == 'red+rgb':
                        set_hsv_trackbar(
                            parameters['red_hsv'], parameters['red_tol'])
                    elif channels[channel][0] == 'blue+rgb':
                        set_hsv_trackbar(
                            parameters['blue_hsv'], parameters['blue_tol'])
                    elif channels[channel][0] == 'mask+rgb':
                        set_hsv_trackbar(
                            parameters['green_hsv'], parameters['green_tol'])

            elif k & 0xFF == ord('0'):  # display rgb
                channel = 0
                logging.info('display channel %d [%s]' % (
                    channel, channels[channel][0]))

            elif k & 0xFF == ord('5'):  # save config

                parameters['filter'] = filter_rgb
                parameters['A4_calib'] = src.tolist()
                parameters['roi'] = roi.tolist()
                parameters['target_hsv'] = [cv2.getTrackbarPos('hue', 'parameters'),
                                            cv2.getTrackbarPos(
                                                'sat', 'parameters'),
                                            cv2.getTrackbarPos('val', 'parameters')]
                parameters['target_tol'] = [cv2.getTrackbarPos('hue_tol', 'parameters'),
                                            cv2.getTrackbarPos(
                                                'sat_tol', 'parameters'),
                                            cv2.getTrackbarPos('val_tol', 'parameters')]
                save_param(parameters, 'config_server.json')
                logging.info('save config')

            elif k & 0xFF == ord('v') or stop_recording:  # record video sequence
                if recording or stop_recording:
                    # stop recording
                    recording = False
                    stop_recording = False
                    enlapsed = time() - t_rec_start
                    logging.info(
                        'stop recording (total rec. time: %f sec)' % enlapsed)
                    video_writer.release()

                else:
                    # start recording
                    video_size = bitmap.shape[:2][-1::-1]
                    logging.info('start recording (max rec. time :%f sec) %s' % (
                        record_max_sec, str(video_size)))
                    # video recorder

                    video_writer = cv2.VideoWriter(
                        "output.avi", fourcc, 20, video_size)

                    t_rec_start = time()
                    recording = True
                    stop_recording = False

            elif k & 0xFF == ord('s'):  # save next image
                saved = False

            elif k & 0xFF == ord('q'):  # break the loop if "q" is keyed
                break

    # main loop -------------------------------------------------------------------

    avg_time = ((time() - t0) / image_count)

    logging.info('last FPS %f' % current_fps)
    logging.info('average time: %f [s/loop]' % avg_time)

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
    logging.info('Profiling data ...\n' + s.getvalue())
