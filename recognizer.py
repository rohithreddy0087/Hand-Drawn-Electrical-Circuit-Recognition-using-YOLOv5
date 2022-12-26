"""
Inference of object detection model
"""

import numpy as np
import torch
import os

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

def detect(img):
    """Runs YOLOv5 model to detect bounding boxes and classes of components present in the circuit

    Args:
        img (numpy array): input image

    Returns:
        det (numpy array): bounding boxes and classes
    """
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

        # resizing image
        img = torch.zeros((3, imgsz, imgsz), device=device)  # init img
        img = letterbox(img, new_shape=imgsz)[0]

        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()
        # 0 - 255 to 0.0 - 1.0
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        det = pred[0]
        # scaling coordinates back to original image size
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
        det = det.cpu().numpy()
    
    return det

