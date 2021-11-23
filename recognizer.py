# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:24:49 2020

@author: Dell
"""

import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import os

def detect(img0):
    path = os.path.dirname(__file__)
    with torch.no_grad():
        weights, imgsz = path+'/yolov5/weights/best.pt', 640
    
        device = select_device('cpu')
        conf_thres = 0.55
        iou_thres = 0.55
        classes = None
        agnostic_nms = False
        augment = False
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    
        
    
    
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names 
    
        img = torch.zeros((3, imgsz, imgsz), device=device)  # init img
        
        img = letterbox(img0, new_shape=imgsz)[0]
    
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    
        det = pred[0]
    
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        det = det.cpu().numpy()
    
    return det

