"""This is a simplistic Path Detector.

    A Path Detector module is a python file that contains a `detect` function
    that is capable of analyzing a binarized image in which non-zero pixels
    indicate road boundaries. The analysis should identify the path (or,
    multiple paths) to follow.

    See the description of the `detect` function below for more details.

    This Simplistic Path Detector implements a first crude idea to path
    detection, and needs ample modification in order to obtain a working
    prototype.

    In this Simplistic Path Detector, a path is detected by sampling a single
    row towards the bottom of the image. Non-zero pixels are identified to
    infer the road between the car center and the road center is calculated and
    used as the path to follow. """

import logging
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator





def Draw(points, shape) : 
    img = np.zeros(shape) 
    for pt in points : 
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
    return img

class Line_System : 
    def __init__(self, Amount_of_line, min_angle, origin, shape) :
        '''
        shape : max values we can take on the two axis from the origin point 
        origin : where the line comes from 
        min_angle : the ange from which the view starts 
        '''
        self.hc, self.wc, _ = shape
        self.origin = origin
        max_angle = np.pi - min_angle 
        offset = (np.pi - (2 * min_angle))/ ( Amount_of_line + 1 ) 
        self.angle_lst = [ min_angle + ( (i + 1) * offset) for i in range(Amount_of_line)] 
        self.mask = 0
        for a in self.angle_lst : 
            pos = (np.int(np.floor(np.cos(a)* 50)),np.int(np.floor(np.sin(a)* 50)))
            pos = self.Change_To_Image_Coordinate(pos[0], pos[1]) 
            if isinstance(self.mask, int) : 
                self.mask = cv2.line(np.zeros(shape), pos, origin, (0, 0, 255))
            else : 
                self.mask += cv2.line(np.zeros(shape), pos , origin, (0, 0, 255))
        
  
        #the angle before whom you can compute r from y, x being constant 
        #after you need to use x while y is constant
        self.beta = np.arctan(self.hc / self.wc) 

    def IsWhite(x, y, Image) : 
        if Image[y, x] >= 200 : 
            return True 
        else : 
            return False 

    def Change_To_Image_Coordinate(self, x,y) : 
        y *= -1 
        return (x + self.origin[0], y + self.origin[1],)

    def Get_Distances(self, Image, step_size) : 
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        dst = [] 
        hc, wc = Image.shape 
        test_Img = np.zeros(((hc, wc, 3)))
        for alpha in self.angle_lst : 
            cos = np.cos(alpha) 
            sin = np.sin(alpha) 
            r = 0

            #compute the limit r according to border and current angle  
            limit = 0 
            if alpha < self.beta or  alpha > np.pi - self.beta : 
                limit = self.wc / cos
            else : 
                limit = self.hc / sin 

            while True : 
                r += step_size 
                x = np.int(r * cos)
                y = np.int(r * sin)
                x, y = self.Change_To_Image_Coordinate(x, y)
                if Line_System.IsWhite(x, y, Image) or r >= limit : 
                    dst.append(r) 
                    test_Img += Draw([(x, y)] ,(hc, wc, 3))
                    break
                 
        return dst , test_Img 


class Memory_Replay: 
    def __init__(self, Max_size) : 
        self.buffer = []
        self.max_size =Max_size
    
    def add(self, memory) : 
        
        if len(self.buffer) >= self.max_size : 
            self.buffer.pop(0) 

        self.buffer.append(memory) 



model = load_model('NN_data/The_Prodige.h5')
model.summary()
ls = 0      
memory = Memory_Replay(20)


def Predict(data) : 

    global model 

    command = model.predict(data)
    #logging.info('\n\n {} \n\n'.format(command[0]))
    command = np.argmax(command[0])

 
    if command == 0 : 
        button = 'a' 
    elif command == 1 : 
        button = 'w' 
    elif command == 2 : 
        button = 'd' 

    return button 

frames = 0 


def detect(mask):
    """This function receives a binarized image in its `mask` argument
    (`mask` is a h x w  numpy array where h is the height and w the width).
    The non-zero pixels in the array encode road boundaries.
    The task of this function is to analyze the mask, and identify the path
    that the car should follow.

    Returns: a tuple (dict, img) where:
      `dict` is a dictionary that is transmitted to the state machine
           it should contain all the information that the state machine
           requires to actuate the identified path(s).
           Implementors are free to encode this information in the dictionary
           in any way that they like. Of course, the state machine needs to
           be written correspondingly to correctly decode the information.g

      `img` is optionally a numpy array (of the same width and height
            as the mask) that visualizes the found path. Used for
            visualization in the viewer only.
    """

    global ls
    global memory
    global frames

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    button = ' '
    
    ##### LSTM #####


    if isinstance(ls, int) : 
        ls = Line_System(7, np.pi / 4, (mask.shape[1] //2,  mask.shape[0] - 20 ), mask.shape  )       
    


    distance, pts = ls.Get_Distances( mask , 2 )
    memory.add(distance)
    if len(memory.buffer) >= 20 :
        logging.info('\n\nPredictinge\n\n')
        data = np.array([memory.buffer]) / 150
        button = Predict(data) 
 

    return ( {"command" : button},  pts.astype(np.uint8))
