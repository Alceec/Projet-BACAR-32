"""This is a template for PathDetectors.

   A Path Detector module is a python file that contains a `detect`
   function that is capable of analyzing a binarized image
   in which non-zero pixels indicate road boundaries. The analysis should
   identify the path (or, multiple paths) to follow.

   See the description of the `detect` function below for more details.

   Please note: implementors are free to add additional (auxiliary) methods and
   variables to the file (called from within the detect() function)
"""

import logging
import numpy as np
import cv2

# Log which path detector is being used. This appears in the output. Useful
# for ensuring that the correct path detector is being used.
logging.info('Template PathDetector has been initialized')


def detect(mask):
    """This method receives a binarized image in its `mask` argument
    (`mask` is a h x w  numpy array where h is the height and w the width).
    The non-zero pixels in the array encode road boundaries.
    The task of this method is to analyze the mask, and identify the path
    that the car should follow.
    This method is called by the framwork upon each capture camera frame.

    Returns: a tuple (dict, img) where:
      `dict` is a dictionary that is transmitted to the state machine
           it should contain all the information that the state machine
           requires to actuate the identified path(s).
           Implementors are free to encode this information in the dictionary
           in any way that they like. Of course, the state machine needs to
           be written correspondingly to correctly decode the information.

      `img` is optionally a numpy array (of the same width and height
            as the mask) that visualizes the found path. Used for
            visualization in the viewer only.

    See path_detector.py in the `example` folder for a simplistic but functioning
    example.
    """
    return (None, None)
