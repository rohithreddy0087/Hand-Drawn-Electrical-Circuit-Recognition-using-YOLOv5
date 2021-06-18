# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:06:34 2020

@author: Dell
"""
import numpy as np

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def mid_point(x1,y1,x2,y2):
    return (x1+x2)/2, (y1+y2)/2

def mapping(dim_matrix,comp_dim,nodes):
    maps = []
    for i in range(dim_matrix.shape[0]):
        dim = dim_matrix[i]
        midx,midy = mid_point(dim[1],dim[0],dim[3],dim[2])
        d = []
        for j in range(comp_dim.shape[0]):
            pntx,pnty = comp_dim[j][0],comp_dim[j][1]
            dist = distance(midx,midy,pntx,pnty)
            d.append((dist,j))
        sort_dist = sorted(d)
        sort_arr = np.array(sort_dist)
        maps.append(sort_arr[0:2,1])
        
    # node_comp_map = []
    # maps_arr = np.array(maps)
    # for i in range(comp_dim.shape[0]):
    #     pntx,pnty = comp_dim[i][0],comp_dim[i][1]
    #     rel = np.where(maps == i)
    #     nc = []
    #     for j in range(nodes.shape[0]):
    #         nx,ny = nodes[j][0],nodes[j][1]
    #         dist = distance(nx,ny,pntx,pnty)
    #         nc.append((dist,j))
    #     sort_dist = sorted(nc)
    #     sort_arr = np.array(sort_dist)
    #     node_comp_map.append(int(sort_arr[0,1]))
    node_comp_map = []
    for i in range(len(maps)):
        con1 = int(maps[i][0])
        con2 = int(maps[i][1])
        con_1x,con_1y = comp_dim[con1][0],comp_dim[con1][1]
        con_2x,con_2y = comp_dim[con2][0],comp_dim[con2][1]
        nc1 = []
        nc2 = []
        for j in range(nodes.shape[0]):
            nx,ny = nodes[j][0],nodes[j][1]
            dist1 = distance(nx,ny,con_1x,con_1y)
            dist2 = distance(nx,ny,con_2x,con_2y)
            nc1.append((dist1,j))
            nc2.append((dist2,j))
        sort_dist1 = sorted(nc1)
        sort_dist2 = sorted(nc2)
        sort_arr1 = np.array(sort_dist1)
        sort_arr2 = np.array(sort_dist2)
        if int(sort_arr1[0,1]) == int(sort_arr2[0,1]):
            min_dist = min(sort_arr1[0,0],sort_arr2[0,0])
            if min_dist == sort_arr1[0,0]:
                node_comp_map.append(int(sort_arr1[0,1]))
                node_comp_map.append(int(sort_arr2[1,1]))
            else:
                node_comp_map.append(int(sort_arr1[1,1]))
                node_comp_map.append(int(sort_arr2[0,1]))
        else:
            node_comp_map.append(int(sort_arr1[0,1]))
            node_comp_map.append(int(sort_arr2[0,1]))
    #     node_comp_map.append(int(sort_arr[0,1]))
        
    count_comp = [0]*5
    for i in range(dim_matrix.shape[0]):
        c1 = int(dim_matrix[i][5])
        count = 0
        for j in range(dim_matrix.shape[0]):
            c2 = int(dim_matrix[j][5])
            if c1 == c2 :
                count = count + 1
        count_comp[c1] = count
    
    count_nodes = [0]*nodes.shape[0]
    hanging_nodes = []
    for i in range(nodes.shape[0]):
        n1 = i
        count = 0
        for j in range(len(node_comp_map)):
            n2 = int(node_comp_map[j])
            if n1 == n2 :
                count = count + 1
        count_nodes[n1] = count
        if count < 2:
            hanging_nodes.append((n1,count))
    
    node_node_map = []
    for i in range(len(hanging_nodes)):
        hn = hanging_nodes[i][0]
        cnt = 2-hanging_nodes[i][1]
        hnx,hny = nodes[hn][0],nodes[hn][1]
        hndist = []
        for j in range(len(hanging_nodes)):
            hn1 = hanging_nodes[j][0]
            nx,ny = nodes[hn1][0],nodes[hn1][1]
            dist = distance(nx,ny,hnx,hny)
            hndist.append((dist,hn1))
        sort_dist = sorted(hndist)
        sort_arr = np.array(sort_dist)
        for k in range(cnt):
            node_node_map.append((hn,int(sort_arr[k+1,1])))
    return maps,node_comp_map,node_node_map