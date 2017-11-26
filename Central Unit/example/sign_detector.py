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
from PIL import Image


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
    (x0, y0, w, h) = bb
    #Image.fromarray( bb, "RGB" ).show() 
    return {'sign': 'STOP', 'x0': x0, 'y0': y0, 'w': w, 'h': h}
