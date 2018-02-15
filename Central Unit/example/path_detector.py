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
import cv2 as cv 


def cvtImage_to_mask(image, thresh) : 
    img = cv.cvtColor( image, cv.COLOR_RGB2GRAY) 
    _, img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY) 
    return img

def get_mid_point( p1, p2) : 
    return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 ) 

def Get_heading(point): 
    point.sort()
    heading = []
    i = 0 
    while i < len(point) - 1:  
        heading.append(get_mid_point(point[i], point[i+ 1]))
        i += 2 
    return heading 

def Is_True_Corner(point, image) : 
    w, h = image.shape
    circle = np.zeros([w, h, 3], np.uint8)
    circle = cv.circle(circle, point, 5, (255, 255, 255), 2) 
    circle_mask = cvtImage_to_mask(circle, 250) 
    inter = cv.bitwise_and(image, image, mask = circle_mask)
    points = cv.goodFeaturesToTrack(inter, 2, 0.01, 5)
    if len(points) == 1 : 
        return True 
    if len(points) == 2 : 
        return False 
    else : 
        raise Exception("Error in true corner detection")


def Sort_Heading( heading , Im_width ) : 
    
    dic = {"left": (-1, -1), "forth": (-1, -1), "right": (-1, -1) }

    if len(heading) == 1 : 
        dic["forth"] = heading[0]
        return dic
        
    for h in heading : 
        x,y = h
        if x < (Im_width/2) - 20 : 
            dic["left" ] = h
        elif x > (Im_width / 2 ) + 20 : 
            dic["right"] = h 
        else : 
            dic["forth"] = h 

    return dic


def Angle_to_Cartesian( angle ) : 
    """return the the unit vector in the direction of the angle """ 
    x = 1 
    if angle > np.pi / 2 : 
        x = -1 
    y = np.tan(angle) * x 
    mag = np.sqrt((y**2) + (x**2)) 
    return (x /mag , y/mag)

def Create_Line_Mask( point1, point2, shape) :
    line = np.zeros(shape, np.uint8) 
    line = cv.line(line, point1, point2, (255, 255, 255), 4, lineType=cv.LINE_AA)
    return cvtImage_to_mask(line, 250) 
    
def Distance(pt1, pt2) : 
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 )


class Lines_System : 
    def __init__( self, origin, view_angle, max_height, max_width, Amount_of_Line ) : 
        '''
        origin : point where lines originate from
        view_angle : width in radians of field of view
        max_height : height of the image 
        max_width : half the width of the image 
        Amount_of_Line : well... the amount of line to draw in the field of view
        '''
        Imprecision_angle_offset = 0.08 # ~ 4 degrees
        self.origin = origin 
        self.max_height = max_height 
        self.max_width = max_width
        max_angle = (np.pi + view_angle) / 2
        min_angle = (np.pi - view_angle) / 2 
        angle_offset = view_angle / (Amount_of_Line + 2) 
        #create mask with the angle for every line 
        self.Mask_ls = []
        for line_idx in range(Amount_of_Line) : 
            current_angle = min_angle + (angle_offset * (line_idx + 1)) + Imprecision_angle_offset
            Unit_vec = Angle_to_Cartesian( current_angle)
            x_edge = (Unit_vec[0] / Unit_vec[1]) * self.max_height
            m = Create_Line_Mask(origin, (int(self.max_width + x_edge), 0), [int(self.max_height), int(self.max_width * 2), 3])
            cv.imshow("m", m) 
            cv.waitKey(0)
            self.Mask_ls.append(m)

    def Sort_Points(Point_list) : 
        temp_l = [] 
        for p in Point_list : 
            temp_l.append(p.ravel) 
        return sorted(temp_l) 

    def Get_Distance(self, path_mask):
        """
        path_mask : inverted black and white image of the path
        return a list of the different distances between the car and the borders of the road
        """
        Dist_ls = [] 
        for l_mask in self.Mask_ls : 
            Intersection_mask = cv.bitwise_and(path_mask, path_mask, mask= l_mask) 
            points = cv.goodFeaturesToTrack(Intersection, 2, 0.01, 3)
            points = np.int0(points)  
            #make sure it's taking the first intersection with the border of the road and not 
            #another road on the other side 
            points = Sort_Points(points) 
            Dist_ls.append(Distance(points[0], points[1]))

        return Dist_ls

    def Get_Full_Mask(self) : 
        Img = 0 
        for ms in self.Mask_ls : 
            if isinstance(Img, int) : 
                Img = ms
            else : 
                Img = Img + ms 
        return Img

def Off_Road(pt, mask) :
    if mask[pt[0], pt[1]] > 240 : 
         return True
    else : 
        return False



LS = None 

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
    global LS

    hc, wc = mask.shape

    #initialize the masks of the lines
    if not isinstance(LS, Lines_System) : 
        LS = Lines_System((int(wc/2), hc), np.pi / 2, hc, int(wc/ 2), 5) 

    #create a mask of a circle 
    black_img = np.zeros([hc, wc, 3], np.uint8 )  
    circle_img = cv.circle(black_img, (int(wc/2), int(hc/2) + 40)  , 50, (255, 255, 255), 2) 
    circle_mask = cvtImage_to_mask(circle_img, 10) 

    #road to white and border to black 
    Inverted_mask = cv.bitwise_not(mask) 

    #create new image where the white pixels are those where the circle overlay the path 
    Intersection = cv.bitwise_and(Inverted_mask, Inverted_mask, mask = circle_mask)     
    Intersection = np.float32(Intersection) 

    #find intersection by using corner detection
    corners = cv.goodFeaturesToTrack(Intersection, 6, 0.01, 10) 

    #check all points and remove false positive 
    true_point = []
    for corner in corners : 
        x, y = corner.ravel()
        if Is_True_Corner((x, y), Intersection ) : 
            true_point.append((x, y))
 
    #get heading from intersections 
    heading = Get_heading(true_point)
    heading_dic = Sort_Heading( heading, wc)  

    #get the distance from the border 
    Dist_ls = LS.Get_Distance(Inverted_mask)
    heading_dic["distances"] = Dist_ls

    #prepare the image to show on the screen 
    Line_mask = LS.Get_Full_Mask()
    returned_Img = cv.cvtColor( circle_mask + Line_mask, cv.COLOR_GRAY2RGB ) 

    heading_dic["off_road"] = Off_Road((100, 200 ), mask)

    return (heading_dic, returned_Img )

