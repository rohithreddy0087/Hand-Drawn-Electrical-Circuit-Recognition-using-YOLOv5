"""
This script contains algorithm for mapping nodes and terminals in given electrical circuit diagram
"""
import numpy as np

def distance(x1,y1,x2,y2):
    """ Computes Euclidean distance

    Args:
        x1 (float): x coordinate of first point
        y1 (float): y coordinate of first point
        x2 (float): x coordinate of second point
        y2 (float): y coordinate of second point

    Returns:
        float: distance between two points
    """
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def mid_point(x1,y1,x2,y2):
    """ Computes mid point

    Args:
        x1 (float): x coordinate of first point
        y1 (float): y coordinate of first point
        x2 (float): x coordinate of second point
        y2 (float): y coordinate of second point

    Returns:
        float: mid point of two points
    """
    return (x1+x2)/2, (y1+y2)/2

def mapping(dim_matrix,comp_dim,nodes):
    """Maps nodes and terminals of an electrical circuit

    Args:
        dim_matrix (numpy array): consists of detected objects along with bounding boxes
        comp_dim (numpy array): coordinates of all terminals 
        nodes (numpy array): coordinates of all nodes

    Returns:
        maps (List): components-terminals map
        node_comp_map (List): nodes-terminals map
        node_node_map (List): node-node map
    """
    # mapping mid point of component to terminals of the components
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
    
    # mapping terminals to the nearest nodes
    node_comp_map = []
    for i ,_ in enumerate(maps):
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
    
    # checking for hanging nodes i.e., which are connected to less than two components    
    count_nodes = [0]*nodes.shape[0]
    hanging_nodes = []
    for i in range(nodes.shape[0]):
        n1 = i
        count = 0
        for j ,_ in enumerate(node_comp_map):
            n2 = int(node_comp_map[j])
            if n1 == n2 :
                count = count + 1
        count_nodes[n1] = count
        if count < 2:
            hanging_nodes.append((n1,count))
    
    # mapping hanging nodes to nearest nodes
    node_node_map = []
    for i ,_ in enumerate(hanging_nodes):
        hn = hanging_nodes[i][0]
        cnt = 2-hanging_nodes[i][1]
        hnx,hny = nodes[hn][0],nodes[hn][1]
        hndist = []
        for j ,_ in enumerate(hanging_nodes):
            hn1 = hanging_nodes[j][0]
            nx,ny = nodes[hn1][0],nodes[hn1][1]
            dist = distance(nx,ny,hnx,hny)
            hndist.append((dist,hn1))
        sort_dist = sorted(hndist)
        sort_arr = np.array(sort_dist)
        for k in range(cnt):
            node_node_map.append((hn,int(sort_arr[k+1,1])))
    return maps,node_comp_map,node_node_map
