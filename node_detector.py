import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from intersections import *
import math 
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from kmeans_cluster import cluster

def node_detector(gray):
    #image = cv2.imread('images/t1.png') 
     
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    #ret,th = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
    img = cv2.GaussianBlur(gray,(9,9),0)
    th = cv2.adaptiveThreshold(img,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    edged = th.astype(np.uint8)
    
    blnk = np.zeros_like(gray)
    blnk = blnk.astype(np.uint8)
    
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # for i in contours:
    #     if cv2.contourArea(i) > 
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
    
    
    lines = cv2.HoughLinesP(blnk, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
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
    
    for line in lines_x:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image1,(x1,y1),(x2,y2),(255,0,0),1)
    for line in lines_y:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image1,(x1,y1),(x2,y2),(0,0,255),1)
    # fig,ax = plt.subplots(nrows=1,ncols = 1)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax.imshow(line_image1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('cc6')
    
    segmented = []
    segmented.append(lines_x)
    segmented.append(lines_y)
    intersections = segmented_intersections(segmented)
    
    for i in intersections:
        for x,y in i:
            cv2.circle(line_image,(x,y), 1, (50,0,0), 8)
    # fig,ax = plt.subplots(nrows=1,ncols = 1)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax.imshow(np.logical_not(line_image),'gray')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('cc7')
    contours, hierarchy = cv2.findContours(line_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    # # A list holds the SSE values for each k
    # feat = np.array(intersections)
    # feat = np.squeeze(feat)
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(feat)
    # sse = []
    tmp = np.where(line_image == 50)
    node_dim_tmp = np.zeros((tmp[0].shape[0],2))
    for i in range(tmp[0].shape[0]):
        node_dim_tmp[i][0] = tmp[0][i]
        node_dim_tmp[i][1] = tmp[1][i]
    node_dim = cluster(node_dim_tmp,len(contours),300)
    
    
    return node_dim
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(gray)
    # plt.subplot(222)
    # plt.imshow(th,'gray')
    # plt.subplot(223)
    # plt.imshow(line_image)
    # plt.subplot(224)
    # plt.imshow(image)
