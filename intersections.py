"""
This script consists of functions needed to find intersections for a given group of lines
"""
import numpy as np

def intersection(line1, line2):
    """Finds intersection between given two lines

    Args:
        line1 (List): end points of horizontal line (x,y coordinates)
        line2 (List): end points of vertical line (x,y coordinates)

    Returns:
        List: intersection point
    """

    x1,y1,x2,y2 = line1[0]
    x3,y3,x4,y4 = line2[0]
    a1 = y2-y1
    b1 = x1-x2
    c1 = a1*x1+b1*y1
    
    a2 = y4-y3
    b2 = x3-x4
    c2 = a2*x3+b2*y3
    
    determinant = a1*b2 - a2*b1
    
    if determinant == 0: 
        return -1
    else:
        x0 = (b2*c1 - b1*c2)/determinant 
        y0 = (a1*c2 - a2*c1)/determinant 
        
        if x0 >= min(x1,x2) and x0 <= max(x1,x2) and y0 >= min(y1,y2) and y0 <= max(y1,y2) and \
            x0 >= min(x3,x4) and x0 <= max(x3,x4) and y0 >= min(y3,y4) and y0 <= max(y4,y4):
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            return [[x0, y0]]
        else:
            return -1

def segmented_intersections(lines):
    """Finds the intersections between groups of lines.
    Args:
        lines (List): lists of horizontal and vertical lines
    Returns:
        intersections(List): List of intersecting points
    """

    intersections = []
    group = lines[0]
    next_group = lines[1]
    for line1 in group:
        for line2 in next_group:
            res = intersection(line1, line2)
            if res != -1:
                intersections.append(res) 

    return intersections