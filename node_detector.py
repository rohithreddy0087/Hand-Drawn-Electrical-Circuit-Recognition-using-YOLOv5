import math 
import cv2 
import numpy as np 
from intersections import segmented_intersections

def node_detector(gray):
    """finds coordinates of junctions or nodes present in components removed image.
    Performs hough transform to find lines in the circuit and then segment them into horizontal or vertical lines.
    Intersections of these lines are returned as nodes

    Args:
        gray (numpy array): components removed image

    Returns:
        node_dim (List): list of node coordinates
    """
    
    img = cv2.GaussianBlur(gray,(9,9),0)
    th = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    edged = th.astype(np.uint8)
    
    blnk = np.zeros_like(gray)
    blnk = blnk.astype(np.uint8)
    
    # drawing contours on a blank image
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blnk = cv2.drawContours(blnk, contours,-1, (255, 0, 0), 3) 
    blnk = blnk ==255
    blnk = blnk.astype(np.uint8)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    line_image = np.zeros_like(gray)
    line_image1 = np.ones((blnk.shape[0],blnk.shape[1],3))*255  # creating a blank to draw lines on
    # using hough tranform to detect lines
    lines = cv2.HoughLinesP(blnk, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    # segmenting lines into horizontal and vertical lines
    lines_x = []
    lines_y = []
    for line_i in lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[0][3]),(line_i[0][0]-line_i[0][2]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            lines_y.append(line_i)
        else:
            lines_x.append(line_i)
    lines_x = sorted(lines_x, key=lambda _line: _line[0][0])
    lines_y = sorted(lines_y, key=lambda _line: _line[0][1])
    
    # drawing lines on a blank image to verify
    # for line in lines_x:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image1,(x1,y1),(x2,y2),(255,0,0),1)
    # for line in lines_y:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image1,(x1,y1),(x2,y2),(0,0,255),1)
    
    segmented = []
    segmented.append(lines_x)
    segmented.append(lines_y)

    # finding intersections between horizontal and vertical lines
    intersections = segmented_intersections(segmented)
    
    # drawing nodes on a blank image to find the node locations
    for i in intersections:
        for x,y in i:
            cv2.circle(line_image,(x,y), 1, (50,0,0), 8)
    
    # using contour detection to find node centers
    contours, hierarchy = cv2.findContours(line_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    node_dim = []
    for i,cntr in enumerate(contours):
        M = cv2.moments(cntr)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        node_dim.append([cY,cX])
    
    return node_dim

