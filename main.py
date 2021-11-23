# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:27:55 2020

@author: Dell
"""

import cv2 
import numpy as np
from matplotlib import pyplot as plt
from recognizer import detect
from node_detector import node_detector
from mapping import *
#if __name__ == "__main__":
def main(img,st):
    #img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    main_img = np.copy(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    comp_removed = np.copy(gray)
    dim_matrix = detect(img)
    a_strings = ["%.2f" % x for x in dim_matrix[0]]
    st.write(f'### Dimension {", ".join(list(a_strings))}')
    classes = ['V','C','D','I','R']
    names = ['Voltage Source', 'Capacitor', 'Diode', 'Inductor', 'Resistor']
    boxes = np.zeros_like(gray)
    boxes1 = np.zeros_like(main_img)
    main_img0 = cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB)
    for i in range(dim_matrix.shape[0]):
        dim = dim_matrix[i]
        start = (int(dim[0]),int(dim[1]))
        end = (int(dim[2]),int(dim[3]))
        a_strings = ["%.2f" % x for x in start]
        st.write(f'### Dimension {", ".join(list(a_strings))}')
        a_strings = ["%.2f" % x for x in end]
        st.write(f'### Dimension {", ".join(list(a_strings))}')
        boxes = cv2.rectangle(boxes, start, end, (255,0,0), 1) 
        boxes1 = cv2.rectangle(main_img0, start, end, (255,0,0), 2) 
        comp_removed[int(round(dim[1])):int(round(dim[3])),int(round(dim[0])):int(round(dim[2]))] = 255
    # fig,ax = plt.subplots(nrows=1,ncols = 2)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax[0].imshow(boxes1,'gray')
    # ax[1].imshow(comp_removed,'gray')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # ax[1].set_xticks([])
    # ax[1].set_yticks([])
    # plt.savefig('cc0')
    nodes = node_detector(comp_removed)
    img = cv2.GaussianBlur(gray,(9,9),0)
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    boxes = boxes == 255
    th = th == 255
    comp_pos1 = np.logical_not(np.logical_not(boxes)+th)
    comp_pos1 = comp_pos1.astype(np.uint8)
    comp_pos = comp_pos1*255
    
    comp_dim_tmp = []
    contours, hierarchy = cv2.findContours(comp_pos,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for i,cntr in enumerate(contours):
        M = cv2.moments(cntr)
        length = cv2.arcLength(cntr,True)
        (cx,cy), r = cv2.minEnclosingCircle(cntr)
        comp_dim_tmp.append([cy,cx,length])
    h = len(comp_dim_tmp) - 2*dim_matrix.shape[0]
    comp_dim_tmp = sorted(comp_dim_tmp, key = lambda x: x[2])[h:]    
    comp_dim = []
    for dim in comp_dim_tmp:
        comp_dim.append([dim[0],dim[1]])
    nodes = np.array(nodes)
    comp_dim = np.array(comp_dim)
    
    # kernel = np.ones((5,5),np.uint8)
    # comp_pos = cv2.dilate(comp_pos,kernel,iterations = 2)
    
    # fig,ax = plt.subplots(nrows=1,ncols = 2)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax[0].imshow(boxes,'gray')
    # ax[1].imshow(th,'gray')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # ax[1].set_xticks([])
    # ax[1].set_yticks([])
    # plt.savefig('cc1')
    
    # fig,ax = plt.subplots(nrows=1,ncols = 2)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax[0].imshow(comp_pos1,'gray')
    # ax[1].imshow(comp_pos,'gray')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # ax[1].set_xticks([])
    # ax[1].set_yticks([])
    # plt.savefig('cc2')
    # tmp = np.where(comp_pos == 255)
    # comp_dim_tmp = np.zeros((tmp[0].shape[0],2))
    # for i in range(tmp[0].shape[0]):
    #     comp_dim_tmp[i][0] = tmp[0][i]
    #     comp_dim_tmp[i][1] = tmp[1][i]
    # comp_dim = cluster(comp_dim_tmp,2*dim_matrix.shape[0],100)
    main_img1 = cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB)
    for y,x in comp_dim:
        cv2.circle(main_img1,(int(x),int(y)), 1, (0,255,0), 5)
    # fig,ax = plt.subplots(nrows=1,ncols = 1)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax.imshow(main_img1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('cc3')
    # main_img2 = cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB)
    # for x,y in nodes:
    #     cv2.circle(main_img2,(int(y),int(x)), 1, (250,0,0), 5)
    # fig,ax = plt.subplots(nrows=1,ncols = 1)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax.imshow(main_img2)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('cc4')
    
    for x,y in nodes:
        cv2.circle(main_img1,(int(y),int(x)), 1, (250,0,0), 5)
    # fig,ax = plt.subplots(nrows=1,ncols = 1)
    # fig.set_figheight(8)
    # fig.set_figwidth(8)
    # ax.imshow(main_img1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.savefig('cc5')
    maps,node_comp_map,node_node_map = mapping(dim_matrix,comp_dim,nodes)
            
    f = open("output.txt", "w")
    result = np.ones_like(main_img)
    result = result * 255
    f.write("Components in the circuit are: \n")
    count_ind = [0]*5
    comp_list = []
    for i in range(dim_matrix.shape[0]):
        cl = int(dim_matrix[i][5])
        dim = dim_matrix[i]
        start = (int(dim[0]),int(dim[1]))
        end = (int(dim[2]),int(dim[3]))
        cv2.rectangle(result, start, end, (255,0,0), 2)
        midx,midy = mid_point(dim[1],dim[0],dim[3],dim[2])
        cv2.putText(result,classes[cl]+str(count_ind[cl]+1), (int(midy),int(midx)),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 0), 1, cv2.LINE_AA)
        f.write(names[cl]+" "+classes[cl]+str(count_ind[cl]+1)+"\n")
        comp_list.append(names[cl]+" "+classes[cl]+str(count_ind[cl]+1))
        count_ind[cl] = count_ind[cl] + 1
        
    f.write("Junctions in the circuit are: \n")
    jns_list = []
    for i in range(nodes.shape[0]):
        f.write("Node N"+str(i+1)+"\n")
        jns_list.append("Junction N"+str(i+1))
        x,y= nodes[i]
        cv2.circle(result,(int(y),int(x)), 1, (0,0,255), 6)
        cv2.putText(result, str(i), (int(y),int(x)),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 0), 1, cv2.LINE_AA) 
    
    f.write("Connections in the circuit are: \n")
    conn_list = []
    count_ind = [0]*5
    for i in range(len(maps)):
        cl = int(dim_matrix[i][5])
        n1 =  node_comp_map[2*i]
        n2 = node_comp_map[2*i+1]
        start1 = (int(round(nodes[n1][1])),int(round(nodes[n1][0])))
        end2 = (int(round(nodes[n2][1])),int(round(nodes[n2][0])))
        end1 = (int(round(comp_dim[int(maps[i][0])][1])),int(round(comp_dim[int(maps[i][0])][0])))
        start2 = (int(round(comp_dim[int(maps[i][1])][1])),int(round(comp_dim[int(maps[i][1])][0])))
        cv2.line(result,start1,end1,(0,0,0),2)
        cv2.line(result,start2,end2,(0,0,0),2)
        f.write(classes[cl]+str(count_ind[cl]+1)+" is between Node"+str(n1)+" and Node"+str(n2)+"\n")
        conn_list.append(classes[cl]+str(count_ind[cl]+1)+" is between Node"+str(n1)+" and Node"+str(n2))
        count_ind[cl] = count_ind[cl] + 1
    
    count_node_ind = [0]*len(node_node_map)
    for i in range(len(node_node_map)):
        n1 = node_node_map[i][0]
        n2 = node_node_map[i][1]
        count = 0
        for j in range(len(node_node_map)):
            if j != i:
                n11 = node_node_map[j][0]
                n21 = node_node_map[j][1]
                if n1 == n21 and n2 == n11:
                    count = count+1+count_node_ind[j]
        count_node_ind[i] = count
    for i in range(len(node_node_map)):
        n1 = node_node_map[i][0]
        n2 = node_node_map[i][1]             
        if count_node_ind[i] < 2:
            f.write("Node"+str(n1)+" and "+"Node"+str(n2)+" are connected"+"\n")
            conn_list.append("Node"+str(n1)+" and "+"Node"+str(n2)+" are connected")
            start = (int(round(nodes[n1][1])),int(round(nodes[n1][0])))
            end = (int(round(nodes[n2][1])),int(round(nodes[n2][0])))
            cv2.line(result,start,end,(0,0,0),2)
    f.close()
    
    plt.imshow(result)
    plt.savefig('recognized_circuit')
    return result, boxes1, main_img1, comp_list, jns_list, conn_list

#main('D:/project/Test set/1.PNG')
