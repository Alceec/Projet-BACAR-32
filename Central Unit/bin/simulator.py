"""
MAIN simulator program

- create ARENA
- insert CAR
- simulate CAMERA
- simulate LIMITS

listen to MQTT messages
produce MQTT messages

"""
__author__ = 'olivier'

import numpy as np

from datetime import datetime
from time import time,sleep
from numpy.linalg import norm
import cv2
import json

import paho.mqtt.client as mqtt
import logging
import argparse

from os import sys, path

# add parent directory of current script to search path
sys.path.append(path.dirname(path.abspath(__file__))+'/..')

from tools.param_tools import read_param,save_param
from tools.compression_tools import bool_to_payload,ts_mask_to_payload
from tools.log_tools import create_logger
from mqtt_config import *

class Arena(object):
    def __init__(self,filename):
        if not path.exists(filename):
            logging.error("Arena does not exist: %s" % path.abspath(filename))
            sys.exit(1)

        self.filename = filename
        ima = cv2.imread(filename)

        if ima.ndim ==3:
            self.bitmap = ima.copy()
        else:
            self.bitmap = ima.copy()[:, :, np.newaxis]

        #
        # self.bitmap = np.zeros((ima.shape[0],ima.shape[1],3),dtype=np.uint8)
        # self.bitmap[ima[:,:,0]>0,:] = 255
        # self.bitmap[ima[:,:,0]==0,:] = 128

        self.size_x,self.size_y,_ =self.bitmap.shape
        self.original = self.bitmap.copy()
        # create display window
        cv2.namedWindow('arena')#,cv2.WINDOW_NORMAL)

    def reset_bitmap(self):
        self.bitmap = self.original.copy()

    def show(self):
        cv2.imshow('arena',self.bitmap)

    def crop (self,xy):
        x0 = int(min(xy[:,0]))
        x1 = int(max(xy[:,0]))
        y0 = int(min(xy[:,1]))
        y1 = int(max(xy[:,1]))
        ix0 = min(max(0,x0),self.size_x-1)
        ix1 = min(max(0,x1),self.size_x-1)
        iy0 = min(max(0,y0),self.size_y-1)
        iy1 = min(max(0,y1),self.size_y-1)

        res = np.zeros((x1-x0,y1-y0),dtype=np.uint8)
        ima = self.bitmap[ix0:ix1,iy0:iy1]
        return ima

class Camera(object):
    def __init__(self,x0,y0,z0,v_angle=np.pi/5,h_angle=np.pi/4,v_dir=np.pi/4,h_dir=0,
                      warp_size=(200,200)):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.v_angle = v_angle
        self.h_angle = h_angle
        self.v_dir = v_dir
        self.h_dir = h_dir
        self.warp_size = warp_size

    def xy0(self,dir_x,dir_y,dir_z):
        d = np.array((dir_x,dir_y,dir_z))
        d = d/norm(d)
        a = - self.z0/d[2]
        x = self.x0 + a*d[0]
        y = self.y0 + a*d[1]
        return (x,y)

    def frustum(self):
        # returns 4 corners, in the xy plane of the camera frustum
        # optical axis
        (x,y) = self.xy0(np.cos(self.h_dir),np.sin(self.h_dir),-np.sin(self.v_dir))
        xy = np.zeros((6,2))
        xy[0,:] = self.xy0(np.cos(self.h_dir),np.sin(self.h_dir)+np.sin(self.h_angle),
                             -np.sin(self.v_dir)+np.sin(self.v_angle))
        xy[1,:] = self.xy0(np.cos(self.h_dir),np.sin(self.h_dir)-np.sin(self.h_angle),
                             -np.sin(self.v_dir)+np.sin(self.v_angle))
        xy[2,:] = self.xy0(np.cos(self.h_dir),np.sin(self.h_dir)+np.sin(self.h_angle),
                             -np.sin(self.v_dir)-np.sin(self.v_angle))
        xy[3,:] = self.xy0(np.cos(self.h_dir),np.sin(self.h_dir)-np.sin(self.h_angle),
                             -np.sin(self.v_dir)-np.sin(self.v_angle))
        xy[4,:] = (self.x0,xy[0,1])
        xy[5,:] = (self.x0,xy[1,1])
        return np.array((x,y)),xy

class Car(object):
    def __init__(self,x0,y0,a0,e=30,L=50,wheel_r=10,wheel_l=10):
        self.pos = np.array((x0,y0)) # current absolute position
        self.angle = a0 # current absolute heading
        self.rel_angle = 0 # relative angle
        self.speed = 0
        self.e = e # distance between wheels
        self.L = L # car length
        self.wheel_r = wheel_r # wheels radii
        self.wheel_l = wheel_l
        self.theta_r0 = 0#-np.pi/16 # angular speed for each wheel
        self.theta_l0 = 0#-np.pi/16
        self.theta_r = 0 # angular speed for each wheel
        self.theta_l = 0
        self.t0 = 0 # initial time
        self.rec = []

        self.camera = Camera(40,10,30,h_dir=np.pi)
        self.iter = 0

    def set_arena(self,arena):
        self.arena = arena

    def draw(self):
        self.element = []

        self.arena.reset_bitmap()

        # opencv transform
        #M = cv2.getRotationMatrix2D((self.pos[0],self.pos[1]),-self.angle*180./np.pi,1)
        # M = cv2.getRotationMatrix2D((0,0),-self.angle*180./np.pi,1)
        # T = np.array(((1,0,self.pos[0]),(0,1,self.pos[1])))
        a = self.angle#*180./np.pi
        R = np.array([[np.cos(a),-np.sin(a),0],
                      [np.sin(a),np.cos(a),0],
                      [0,0,1]])
        T = np.array([[1,0,self.pos[0]],
                      [0,1,self.pos[1]],
                      [0,0,1]])
        M = np.dot(T,R)

        def transform(xy,M):
            tmp = np.asarray(xy)
            tmp = np.hstack((tmp,np.ones((tmp.shape[0],1))))
            return np.dot(M,tmp.T).T[:,:2]

        def cv_draw_poly(bitmap,xy,M,color=(0,0,0),closed=True):
            pxy = transform(xy,M)
            cv2.polylines(bitmap,[np.int32(pxy)],closed,color)

        def cv_draw_rect(bitmap,p,w,h,M,color=(0,0,0)):
            xy = [(p[0]-w,p[1]-h),
                 (p[0]-w,p[1]+h),
                 (p[0]+w,p[1]+h),
                 (p[0]+w,p[1]-h)]
            cv_draw_poly(bitmap,xy,M,color)


        #car
        cv_draw_rect(self.arena.bitmap,(0,0),2,2,M)
        cv_draw_rect(self.arena.bitmap,(self.L/2,0),self.L/2,self.e/2,M,color=(255,50,50))
        cv_draw_rect(self.arena.bitmap,(0,self.e/2+2),self.wheel_r*2,2,M)
        cv_draw_rect(self.arena.bitmap,(0,-self.e/2-2),self.wheel_l*2,2,M)
        # camera
        cv_draw_rect(self.arena.bitmap,(self.camera.x0-1,self.camera.y0-1),2,2,M)
        # draw frustum
        xy_c,xy_f = self.camera.frustum()
        cv_draw_rect(self.arena.bitmap,(xy_c[0]-1,xy_c[1]-1),2,2,M)
        cv_draw_poly(self.arena.bitmap,xy_f[[0,1,3,2],:],M,color=(0,0,255))
        cv_draw_poly(self.arena.bitmap,xy_f[[0,1,5,4],:],M,color=(0,0,255))
        if self.rec:
            xy = np.asarray(self.rec)
            cv_draw_poly(self.arena.bitmap,xy,np.eye(3,3),closed=False,color=(50,50,50))

        # get coordinates transformed of the frustum
        XY_f = transform(xy_f,M)

        scale = .7

        src = XY_f[[0,1,5,4],:].astype(np.float32)
        dst = scale * (xy_f[[0,1,5,4],:].astype(np.float32) - np.asarray([min(xy_f[[0,1,5,4],0]),min(xy_f[[0,1,5,4],1])],dtype=np.float32))

        M = cv2.getPerspectiveTransform(src,dst)

        # frustum trace
        p = xy_f[:,:].astype(np.float32) - np.asarray([min(xy_f[[0,1,5,4],0]),min(xy_f[[0,1,5,4],1])],dtype=np.float32)

        dx = max(dst[:,0])#-min(dst[:,0])
        dy = max(dst[:,1])#-min(dst[:,1])

        cv2.imshow('arena',self.arena.bitmap)

        self.arena.reset_bitmap() #remove car lines before cropping camera view
        warped = cv2.warpPerspective(self.arena.bitmap, M, (int(dx),int(dy)))
        warped = np.transpose(warped,axes=[1,0,2])
        warped = warped[:,-1::-1,0]
        #warped[int(p[2,0]),int(p[2,1])] = 0
        #warped[int(p[3,0]),int(p[3,1])] = 0
        warped = warped.copy()
        cv2.fillPoly(warped, [np.int32(scale*p[[0,4,5,1,3,2,0],-1::-1])], (255,255,255))


        if self.t0 > 0:
            # MQTT send bool
            tmp = warped == 255
            t = time()
            payload = ts_mask_to_payload(t, tmp)
            #payload = bool_to_payload(tmp[:,:,0])
            self.t0 = 0
            self.client.publish(MSG_BOOL_ARRAY, payload)
        else:
            self.t0 += 1



    def update(self):

        self.rec.append(self.pos)
        if len(self.rec) > 100:
            self.rec.pop(0)

        #vl = self.theta_l * self.wheel_l
        #vr = self.theta_r * self.wheel_r
        #v = .5 * (vl+vr)
        #d_angle = (vl - vr)/self.e

        #pos = self.pos + v * np.array((np.cos(self.angle),np.sin(self.angle)))
        self.angle = self.angle + np.pi/180 *  self.rel_angle

        pos = self.pos + self.speed * np.array((np.cos(self.angle),np.sin(self.angle)))

        pos[0] = min(self.arena.size_x,max(0,pos[0]))
        pos[1] = min(self.arena.size_y,max(0,pos[1]))
        self.pos = pos


        self.remove()
        self.draw()




    def remove(self):
        for e in self.element:
            e.remove()




# MQTT EVENTS
# --------------------------------------------------------------------------------------------------------------
def on_connect(client, userdata, flags, rc):
    # callback function called after connectionhas been established with the MQTT broker
    logging.info("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    #client.subscribe("$SYS/#")
    client.subscribe(MSG_PATH_JSON)
    # listen to what is communicated on the MSG_DRIVER_SEND_JSON topic
    client.subscribe(MSG_DRIVER_SEND_JSON)

def on_message(client, userdata, msg):
    # callback function called after a message is received from the MQTT broker
    if msg.topic[0]=='$': # skip service messages
        pass
        #print(msg.topic)
    if msg.topic == MSG_DRIVER_SEND_JSON: # change car behaviour
        d = json.loads(msg.payload.decode('utf-8'))
        #logging.info('json %s'%str(d))
        # d is a dict with keys x,y,u,v where x and y have integer payloads
        # and u,v, have float payloads. interpret u as speed and v as angle
        userdata.speed = -d['u'] #speed and angle need to be inverted since our origin is at (0,0) and the car center is at (x, y) with y positive facing towards (0,0)
        userdata.rel_angle = -d['v']



# --------------------------------------------------------------------------------------------------------------


def get_arguments():
    parser = argparse.ArgumentParser(description='Simulator')
    parser.add_argument('--arena', nargs='?', default=None,
                        help='specify the name of the arena (default: specified in config file; key=arena).')
    parser.add_argument('--config', nargs='?', default=None,
                        help='specify config file to use (default: ./config_server.json)')

    return parser.parse_args()


def get_parameters(args):
    paramsfile = 'config_server.json'
    if args.config is not None:
        paramsfile = args.config

    params = read_param(paramsfile)
    # default parameters
    default_params = {'mqtt_host': 'localhost',
                      'mqtt_port': 1883,
                      'profile': False,
                      'arena': './data/test_arena9s.png'}

    if params is None:
        params = default_params
        logging.warning("Default parameters will be used")
    else:
        # config file overrides defaults
        p = default_params.copy()
        p.update(params)
        params = p

    # arguments passed on command line override everything else
    if args.arena is not None:
        params['arena'] = args.arena  # command line overrides config file
    return params


def setup_logging():
    log_file_name = "./%s.log" % path.basename(__file__)
    scriptname = "%s" % path.basename(__file__)
    print("Logging to " + path.abspath(log_file_name))
    create_logger(log_file_name, scriptname=scriptname)
    # output python version
    logging.info("Running on python version" + sys.version.replace("\n", "\t"))


if __name__ == '__main__':

    #start logging
    setup_logging()

    args = get_arguments()

    # read parameters file
    parameters = get_parameters(args)

    if parameters is None:
        logging.error('No parameters found')
        sys.exit(1)

    # open MQTT connection
    client = mqtt.Client()
#    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(parameters['mqtt_host'], parameters['mqtt_port'], 10)

    arena = Arena(parameters['arena'])
    car = Car(300,256,+np.pi/2)

    # perspective parameters
#    WARP_IMAGE_SIZE = tuple(parameters['warp_size'])
#    offset = np.array(parameters['offset'],dtype=np.float32)
#    car.camera.warp_size = WARP_IMAGE_SIZE

    car.set_arena(arena)
    car.client = client

    car.theta_r = car.theta_r0
    car.theta_l = car.theta_l0

    client.user_data_set(car)

    arena.show()
    car.draw()

    try:
        while True:
            # Wait for key (needed to display image)
            k = cv2.waitKey(1)

            client.loop(.05)
            car.update()
            sleep(.05)

            if k & 0xFF == ord('h'): #hide/unhide realtime image
                hide = not hide
                logging.info("hide is %s"%str(hide))
            if k & 0xFF == ord('q'): # break the loop if "q" is keyed
                break
    except KeyboardInterrupt:
        pass
    except:
        logging.exception("Exception occurred:")
        sys.exit(1)
