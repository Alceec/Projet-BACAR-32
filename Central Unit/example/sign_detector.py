"""This is a Simplistic Sign Detector modules.

   A Sign Detector module is a python file that contains a `detect` function
   that is capable of analyzing a color image which, according to the image
   server, likely contains a road sign. The analysis should identify the kind
   of road sign contained in the image.

   See the description of the `detect` function below for more details.
"""

import logging
import numpy as np
import cv2
from scipy import ndimage

# Log which sign detector is being used. This appears in the output. Useful
# for ensuring that the correct sign detector is being used.
logging.info('Simplistic SignDetector has been initialized')

def detect(bb, sign):
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
    - None, if the sign is determined to be none of the above

    Returns: a dictionary dict that contains information about the recognized
    sign. This dict is transmitted to the state machine it should contain
    all the information that the state machine to act upon the sign (e.g.,
    the type of sign, estimated distance).

    This simplistic detector always returns "STOP", copies the bounding box
    to the dictionary.
    """
    
    x0, y0, w, h = bb
    res = None

    if y0 > 55 and y0 < 90 and x0 > 195 and x0 <= 290:      
        #Lecture de l'image
        #sign = cv2.imread(sign)
        #cv2.imshow("sign", sign)
        lower_color_r1 = np.array([0, 70, 60])
        upper_color_r1 = np.array([15, 255, 255])
        lower_color_r2 = np.array([165, 70, 60])
        upper_color_r2 = np.array([179, 255, 255])
        hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, lower_color_r1, upper_color_r1)
        mask_r2 = cv2.inRange(hsv, lower_color_r2, upper_color_r2)
        mask_r = mask_r1 + mask_r2
        #cv2.imshow("mask_r", mask_r)
        number_of_columns = np.shape(hsv)[1]
        number_of_rows = np.shape(hsv)[0]
        image_size = np.size(mask_r)
        #plt.subplot(231)
        #plt.imshow(mask_r)
        #mask_r = cv2.blur(mask_r, (int(number_of_columns/10),int(number_of_columns/10)))
        #_, mask_r = cv2.threshold(mask_r, 140, 255, cv2.THRESH_BINARY)
        #t_r = np.where(mask_r == 255)
        percent_stop_v, percent_stop_h = 0, 0
        
        if cv2.countNonZero(mask_r) > image_size/4:
            #logging.info('1') #Entering stop detection
            """bord_haut = t_r[0].min()
            bord_bas = t_r[0].max()
            bord_gauche = t_r[1].min()
            bord_droit = t_r[1].max()
            mask_r = mask_r[bord_haut:bord_bas, bord_gauche:bord_droit]"""
            #cv2.imshow("mask_r*", mask_r)
            number_of_columns = np.shape(mask_r)[1]
            number_of_rows = np.shape(mask_r)[0]
            percent_stop_v = cv2.countNonZero(mask_r[:, int(number_of_columns*0.5)])/number_of_rows*100
            percent_stop_h = cv2.countNonZero(mask_r[int(number_of_rows*0.5), :])/number_of_columns*100
            #print("V :",percent_stop_v,'%',"\nH :", percent_stop_h,'%')
            """
            plt.subplot(233)
            plt.imshow(mask_r)
            plt.subplot(232)
            plt.imshow(sign)    
            cv2.imshow("sign", mask_r)
            cv2.waitKey()
            cv2.destroyAllWindows()
            """
            
            if percent_stop_v > 45 and percent_stop_h < 15:
                res = "STOP"
                if x0 > 240 or y0 > 70:
                    res = "STOP!"
                
        
        else:
            lower_color_b = np.array([100, 145, 40])
            upper_color_b = np.array([120, 255, 255])
            mask_b = cv2.inRange(hsv, lower_color_b, upper_color_b)
            #plt.subplot(234)
            #plt.imshow(mask_b)
            #mask_b = cv2.blur(mask_b, (int(number_of_columns/10),int(number_of_columns/10)))
            #_, mask_b = cv2.threshold(mask_b, 140, 255, cv2.THRESH_BINARY)
            #cv2.imshow("mask_b", mask_b)
            #t_b = np.where(mask_b == 255)
            percent_blue_v, percent_blue_h = 0, 0
            center_of_massxb, center_of_massxw = 0, 0
            
            if cv2.countNonZero(mask_b) > image_size/3.5:
                #logging.info((x0, y0))
                #logging.info('2') #Entering turn detection
                """bord_haut = t_b[0].min()
                bord_bas = t_b[0].max()
                bord_gauche = t_b[1].min()
                bord_droit = t_b[1].max()
                mask_b = mask_b[bord_haut:bord_bas, bord_gauche:bord_droit]"""
                mask_b_arrow = mask_b[int(number_of_rows*(0.4)):int(number_of_rows*(0.70)),:]
                cv2.imshow("mask_b*", mask_b_arrow)
                lower_color_w = np.array([0, 0, 100])
                upper_color_w = np.array([179, 90, 255])
                mask_w = cv2.inRange(hsv, lower_color_w, upper_color_w)
                mask_w_arrow = mask_w[int(number_of_rows*(0.4)):int(number_of_rows*(0.70)),:]
                cv2.imshow("mask_w", mask_w)#_arrow)
                mask_w_arrow = cv2.blur(mask_w_arrow, (int(number_of_columns/5),int(number_of_columns/5)))
                _, mask_w_arrow = cv2.threshold(mask_w_arrow, 180, 255, cv2.THRESH_BINARY)
                
                if cv2.countNonZero(mask_w_arrow) != 0:
                    #cv2.imshow("mask_w*", mask_w_arrow)
                    _, center_of_massxb = ndimage.measurements.center_of_mass(mask_b)
                    _, center_of_massxw = ndimage.measurements.center_of_mass(mask_w_arrow)
                    
                    if center_of_massxw < number_of_columns/2 and center_of_massxb > number_of_columns/2:
                        test2 = "LEFT"           #res = "LEFT"
                    
                    elif center_of_massxw > number_of_columns/2 and center_of_massxb < number_of_columns/2:
                        test2 = "RIGHT"          #res = "RIGHT"
                    else:
                        test2 = None
                    number_of_columns = np.shape(mask_b)[1]
                    number_of_rows = np.shape(mask_b)[0]
                    percent_blue_v = cv2.countNonZero(mask_b[:, int(number_of_columns*(0.4))])/number_of_rows*100
                    percent_blue_h = cv2.countNonZero(mask_b[int(number_of_rows*(0.525)), :])/number_of_columns*100
                    #print("V :",percent_blue_v,'%',"\nH :", percent_blue_h,'%')
                    

                
                    
                    if percent_blue_v > 35 and percent_blue_h < 22:
                        test1 = "LEFT"
                        if percent_blue_v > 60:
                            test1 = "RIGHT"
                    else:
                        test1 = None
                    
                    if test1 == test2:
                        res = test1
                    elif test1 == None and test2 != None:
                        res = test2
                    elif test2 == None and test1 != None:
                        res = test1
                    
                    #logging.info(test1)
                    #logging.info(percent_blue_v)
            
                    """
                    plt.subplot(236)
                    plt.imshow(mask_b)
                    plt.subplot(235)
                    plt.imshow(sign)    
                    cv2.imshow("sign", mask_b)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
                    """

    logging.info(res)
    return {'sign': res, 'x0': x0, 'y0': y0, 'width': w, 'height': h}

