from scipy import ndimage
import numpy as np
import cv2
import logging

def profile(ima,p0,p1,num):
    n = np.linspace(p0[0],p1[0],num)
    m = np.linspace(p0[1],p1[1],num)
    return [n,m,ndimage.map_coordinates(ima, [m,n], order=0)]

def profile_mn(ima,m,n):
    return [ndimage.map_coordinates(ima, [m,n], order=0)]

def cyclic_inRange(hsv,lower,upper):
    # in range function adapted for hue image (hue is a cyclic value)

    if lower[0]<upper[0]:
        return cv2.inRange(hsv,lower,upper)
    else:
        lower1 = np.array([0,lower[1],lower[2]])
        upper1 = np.array([upper[0],upper[1],upper[2]])
        lower2 = np.array([lower[0],lower[1],lower[2]])
        upper2 = np.array([179,upper[1],upper[2]])
        return cv2.bitwise_or(cv2.inRange(hsv,lower1,upper1),cv2.inRange(hsv,lower2,upper2))

def rot90(image,mode):
    # 90 deg rotation inplace
    # mode: 1=CW, 2=CCW, 3=180

  if mode==1:
    res = cv2.transpose(image)
    res = cv2.flip(res, 1) #transpose+flip(1)=CW
  elif mode==2:
    res = cv2.transpose(image)
    res = cv2.flip(res,0) #transpose+flip(0)=CCW
  elif mode==3:
    res = cv2.flip(image,-1)    #flip(-1)=180
  else:
    logging.error("Unknown rotation flag(",mode)
  return res
