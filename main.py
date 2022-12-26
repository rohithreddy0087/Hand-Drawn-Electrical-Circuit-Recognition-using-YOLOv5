
import cv2 
import numpy as np
from recognizer import detect
from node_detector import node_detector
from mapping import mapping, mid_point

def main(img):
    """main function where all algorithms are called

    Args:
        img (numpy array): input image

    Returns:
        result (numpy array): final rebuilt image
        boxes1 (numpy array): bounding boxes on given input image
        main_img1 (numpy array): nodes and terminals on given input image
        comp_list (List): list of all the components detected 
        jns_list (List): list of all the junctions detected 
        conn_list (List): list of connections traced 
    """
    # converting image to grayscale
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    main_img = np.copy(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    comp_removed = np.copy(gray)

    # running object detection model on image
    dim_matrix = detect(img)
    classes = ['V','C','D','I','R']
    names = ['Voltage Source', 'Capacitor', 'Diode', 'Inductor', 'Resistor']

    # making a copy of the image, plotting bboxes on image and removing detected bounding boxes
    boxes = np.zeros_like(gray)
    boxes1 = np.zeros_like(main_img)
    main_img0 = cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB)
    for i in range(dim_matrix.shape[0]):
        dim = dim_matrix[i]
        start = (int(dim[0]),int(dim[1]))
        end = (int(dim[2]),int(dim[3]))
        boxes = cv2.rectangle(boxes, start, end, (255,0,0), 1) 
        boxes1 = cv2.rectangle(main_img0, start, end, (255,0,0), 2) 
        comp_removed[int(round(dim[1])):int(round(dim[3])),int(round(dim[0])):int(round(dim[2]))] = 255
    
    # detecting nodes on the components removed image
    nodes = node_detector(comp_removed)

    # detecting terminals using bboxes image and thresholded input image
    img = cv2.GaussianBlur(gray,(9,9),0)
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    boxes = boxes == 255
    th = th == 255
    comp_pos1 = np.logical_not(np.logical_not(boxes)+th)
    comp_pos1 = comp_pos1.astype(np.uint8)
    comp_pos = comp_pos1*255
    comp_dim_tmp = []

    # using contour detection to find the exact centers of the terminals
    contours, hierarchy = cv2.findContours(comp_pos,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for i,cntr in enumerate(contours):
        M = cv2.moments(cntr)
        length = cv2.arcLength(cntr,True)
        (cx,cy), r = cv2.minEnclosingCircle(cntr)
        comp_dim_tmp.append([cy,cx,length])
    h = len(comp_dim_tmp) - 2*dim_matrix.shape[0]
    comp_dim_tmp = sorted(comp_dim_tmp, key = lambda x: x[2])#[h:]    
    comp_dim = []
    for dim in comp_dim_tmp:
        comp_dim.append([dim[0],dim[1]])
    nodes = np.array(nodes)
    comp_dim = np.array(comp_dim)
    
    # drawing terminals and nodes on the input image, to verify
    main_img1 = cv2.cvtColor(main_img,cv2.COLOR_BGR2RGB)
    for y,x in comp_dim:
        cv2.circle(main_img1,(int(x),int(y)), 1, (0,255,0), 5)
    for x,y in nodes:
        cv2.circle(main_img1,(int(y),int(x)), 1, (250,0,0), 5)

    # mapping nodes, terminals and components
    maps,node_comp_map,node_node_map = mapping(dim_matrix,comp_dim,nodes)
    
    # generating .txt file with components and connections
    f = open("output.txt", "w")

    # rebuilding circuit on a plain image
    result = np.ones_like(main_img)
    result = result * 255

    # writes all the components present and draws them on a plain image
    f.write("Components in the circuit are: \n")
    count_ind = [0]*len(classes)
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
    
    # writes all the nodes/junctions present and draws them on a plain image
    f.write("Junctions in the circuit are: \n")
    jns_list = []
    for i in range(nodes.shape[0]):
        f.write("Node N"+str(i+1)+"\n")
        jns_list.append("Junction N"+str(i+1))
        x,y= nodes[i]
        cv2.circle(result,(int(y),int(x)), 1, (0,0,255), 6)
        cv2.putText(result, str(i), (int(y),int(x)),cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 0), 1, cv2.LINE_AA) 
    
    # writes all the connections present and draws them on a plain image
    f.write("Connections in the circuit are: \n")
    conn_list = []
    count_ind = [0]*len(classes)
    for i,_ in enumerate(maps):
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
    for i,_ in enumerate(node_node_map):
        n1 = node_node_map[i][0]
        n2 = node_node_map[i][1]
        count = 0
        for j,_ in enumerate(node_node_map):
            if j != i:
                n11 = node_node_map[j][0]
                n21 = node_node_map[j][1]
                if n1 == n21 and n2 == n11:
                    count = count+1+count_node_ind[j]
        count_node_ind[i] = count
    for i ,_ in enumerate(node_node_map):
        n1 = node_node_map[i][0]
        n2 = node_node_map[i][1]             
        if count_node_ind[i] < 2:
            f.write("Node"+str(n1)+" and "+"Node"+str(n2)+" are connected"+"\n")
            conn_list.append("Node"+str(n1)+" and "+"Node"+str(n2)+" are connected")
            start = (int(round(nodes[n1][1])),int(round(nodes[n1][0])))
            end = (int(round(nodes[n2][1])),int(round(nodes[n2][0])))
            cv2.line(result,start,end,(0,0,0),2)
    f.close()
    return result, boxes1, main_img1, comp_list, jns_list, conn_list