"""This is a Simplistic Sign Detector modules.
   A Sign Detector module is a python file that contains a `detect` function
   that is capable of analyzing a color image which, according to the image
   server, likely contains a road sign. The analysis should identify the kind
   of road sign contained in the image.
   
   See the description of the `detect` function below for more details."""

import logging
import numpy as np
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt

# Log which sign detector is being used. This appears in the output. Useful
# for ensuring that the correct sign detector is being used.
logging.info('Simplistic SignDetector has been initialized')

def detect(data, sign):
    """This method receives:
    - sign: a color image (numpy array of shape (h,w,3))
    - bb which is the bounding box of the sign in the original camera view
      bb = (x0,y0, w, h) where w and h are the widht and height of the sign
      (can be used to determine e.g., whether the sign is to the left or
       right)
       
    The goal of this function is to recognize  which of the following signs
    it really is:
    - a stop sign
    - a turn left sign
    - a turn right sign
    - None, if the sign is determined to be none of the 
    
    Returns: a dictionary dict that contains information about the recognized
    sign. This dict is transmitted to the state machine it should contain
    all the information that the state machine to act upon the sign (e.g.,
    the type of sign, estimated distance).
    This simplistic detector always returns "STOP", copies the bounding box
    to the dictionary."""

 
    x0, y0 = data[0:2]
    w, h = data[2:4]
    res = None
    
    if y0 > 60 and y0+h < 120 and x0 > 195 and x0+w <= 310:      
        lower_color_r1 = np.array([0, 70, 60])
        upper_color_r1 = np.array([15, 255, 255])
        lower_color_r2 = np.array([165, 70, 60])
        upper_color_r2 = np.array([179, 255, 255])
        hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, lower_color_r1, upper_color_r1)
        mask_r2 = cv2.inRange(hsv, lower_color_r2, upper_color_r2)
        mask_r = mask_r1 + mask_r2
        image_size = np.size(mask_r)
        percent_stop_v, percent_stop_h = 0, 0
        if cv2.countNonZero(mask_r) > image_size/4:
            percent_stop_v = cv2.countNonZero(mask_r[:, int(w*0.5)])/h*100
            percent_stop_h = cv2.countNonZero(mask_r[int(h*0.5), :])/w*100
            if percent_stop_v > 2*percent_stop_h and percent_stop_v > 40:
                res = "STOP"
                if x0 > 240 or y0 > 70:
                    res = "STOP!"
                    
        else:
            lower_color_b = np.array([100, 145, 40])
            upper_color_b = np.array([120, 255, 255])
            mask_b = cv2.inRange(hsv, lower_color_b, upper_color_b)
            center_of_massxb, center_of_massxw = 0, 0
            if cv2.countNonZero(mask_b) > image_size/3.5:
                lower_color_w = np.array([0, 0, 100])
                upper_color_w = np.array([179, 90, 255])
                mask_w = cv2.inRange(hsv, lower_color_w, upper_color_w)
                mask_w_arrow = mask_w[int(h*(0.4)):int(h*(0.7)),:]
                mask_w_arrow = cv2.blur(mask_w_arrow, (int(w/5),int(w/5)))
                _, mask_w_arrow = cv2.threshold(mask_w_arrow, 220, 255, \
                cv2.THRESH_BINARY)
                if cv2.countNonZero(mask_w_arrow) != 0:
                    center_of_massyb, center_of_massxb = \
                        ndimage.measurements.center_of_mass(mask_b)
                    center_of_massyw, center_of_massxw = \
                        ndimage.measurements.center_of_mass(mask_w_arrow)
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    if center_of_massxw < w/2 and \
                    center_of_massxw < center_of_massxb:
                        res = "LEFT"
                    elif center_of_massxw > w/2 and \
                    center_of_massxw > center_of_massxb:
                        res = "RIGHT"
                    else:
                        res = None

    return {'sign': res, 'x0': x0, 'y0': y0, 'width': w, 'height': h}