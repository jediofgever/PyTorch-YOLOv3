from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator



import json
import datetime
 
 


import os, os.path
import sys
import random
import math
import re
import time
import numpy as np
 
import cv2
 
import sys
import time

# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from vision_msgs.msg import BoundingBox2D


from cv_bridge import CvBridge, CvBridgeError
 
import PIL
import time

imsize = 416
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image):
    """load image, returns cuda tensor"""

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = PIL.Image.fromarray(img)
    image = loader(im_pil).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

 

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--model_def", type=str, default="/home/atas/catkin_ws/catkin_ws_py3_nn/src/ROS_NNs_FANUC_LRMATE200ID/PyTorch-YOLOv3/config/yolov3-custom.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="/home/atas/catkin_ws/catkin_ws_py3_nn/src/ROS_NNs_FANUC_LRMATE200ID/PyTorch-YOLOv3/weights/yolov3_ckpt_98.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="/home/atas/catkin_ws/catkin_ws_py3_nn/src/ROS_NNs_FANUC_LRMATE200ID/PyTorch-YOLOv3/data/custom/classes.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=960, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
print(opt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("output", exist_ok=True)
# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode
classes = load_classes(opt.class_path)  # Extracts class labels from file
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

 
class YOLO3_ROS_Node:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/maskrcnn/segmented",
                                         Image)

        self.yolo_detection_pub = rospy.Publisher("/yolo/detection_boxes",
                                         Detection2DArray)

        self.subscriber = rospy.Subscriber("/camera/color/image_raw",
                                           Image, self.callback, queue_size=1, buff_size=2002428800)
 
        self.bridge = CvBridge()                                                         
        self.counter = 2000
        self.start_time = time.time()
        self.x = 1 # displays the frame rate every 1 second
 
    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and OBJECTS detected'''
        #### direct conversion to CV2 ####
        original_img = self.bridge.imgmsg_to_cv2(ros_data, desired_encoding="bgr8")
        print(original_img.shape)
        to_square = original_img.shape[1] - original_img.shape[0]
        cv_image = cv2.copyMakeBorder(original_img, 0, to_square, 0, 0, cv2.BORDER_CONSTANT)
        
        #
        #cv_image =  cv2.resize(original_img, (640,640), interpolation = cv2.INTER_AREA)        
        #Uncomment thefollowing block in order to collect training data
        
        #cv2.imwrite("/home/atas/MASKRCNN_REAL_DATASET/"+str(self.counter)+".png",original_img)
        
        #self.counter = self.counter +1 
        #sec = input('PRESS KEY FOR NEXT.\n')
 
        cuda_tensor_of_original_image = image_loader(cv_image)
       # Get detections
        with torch.no_grad():
            detections = model(cuda_tensor_of_original_image)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        
        detection_array = Detection2DArray()

        if len(detections)>0:
            # Rescale boxes to original image
            #detections = rescale_boxes(detections, opt.img_size, cv_image.shape[:2])
            # print(detections)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:

                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                k = original_img.shape[1] / imsize

                # Create a Rectangle patch
                cv2.rectangle(original_img, (x1*k,y1*k), (x2*k,y2*k), (random.randint(
                0, 255), random.randint(0, 255), 55), 2)
                bbx = BoundingBox2D()
                bbx.center.x  = (x1+x2)/2 * k 
                bbx.center.y  = (y1+y2)/2 * k 
                bbx.center.theta = 0
                bbx.size_x = (x2-x1) * k
                bbx.size_y = (y2-y1) * k
                
                
                bbx_for_this_detection = Detection2D()
                bbx_for_this_detection.header.stamp = rospy.Time.now()
                bbx_for_this_detection.bbox = bbx
                detection_array.detections.append(bbx_for_this_detection)
                detection_array.header.stamp = rospy.Time.now()


        self.yolo_detection_pub.publish(detection_array)  
                # Add the bbox to the plot
                # Add label
        #### PUBLISH SEGMENTED IMAGE ####
        msg = self.bridge.cv2_to_imgmsg(original_img, "bgr8")
        msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(msg)
        self.counter+=1
        if (time.time() - self.start_time) > self.x :
            print("FPS: ", self.counter / (time.time() - self.start_time))
            self.counter = 0
            self.start_time = time.time()    
         
# Run Node
if __name__ == '__main__':
    '''Initializes and cleanup ros node'''
    ic = YOLO3_ROS_Node()
    rospy.init_node('YOLO3_ROS_Node', anonymous=True)
    rospy.Rate(30)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
