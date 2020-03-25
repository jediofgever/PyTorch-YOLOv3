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


from motor_part.maskrcnn_train import  MotorPartConfig
from mrcnn.config import Config
import json
import datetime
from mrcnn.model import log
import mrcnn.model as modellib
import mrcnn.utils as utils
from skimage.io import imsave, imread

from mrcnn import utils
import os, os.path
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import skimage

import cv2
import h5py
import sys
import time

# Ros libraries
import roslib
import rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.util import img_as_float
import PIL

imsize = 640
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
parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_99.pth", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=640, help="size of each image dimension")
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

        self.subscriber = rospy.Subscriber("/camera/color/image_raw",
                                           Image, self.callback, queue_size=1, buff_size=2002428800)
        self.bridge = CvBridge()                                                         
        self.counter = 1200


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and OBJECTS detected'''
        #### direct conversion to CV2 ####
        cv_image = self.bridge.imgmsg_to_cv2(ros_data, desired_encoding="bgr8")
        cv_image = cv2.resize(cv_image, (640,360), interpolation = cv2.INTER_AREA)
        cv_image = cv2.copyMakeBorder(cv_image, 0, 280, 0, 0, cv2.BORDER_CONSTANT) 

        # Uncomment thefollowing block in order to collect training data
        '''
        cv2.imwrite("/home/atas/MASKRCNN_REAL_DATASET/"+str(self.counter)+".png",cv_image)
        self.counter = self.counter +1 
        '''
   
        cuda_tensor_of_original_image = image_loader(cv_image)
       # Get detections
        with torch.no_grad():
            detections = model(cuda_tensor_of_original_image)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)




        if detections is not None:
            # Rescale boxes to original image
            #detections = rescale_boxes(detections, opt.img_size, cv_image.shape[:2])
            print(detections)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                # Create a Rectangle patch
                cv2.rectangle(cv_image, (x1,y1), (x2,y2), (255,0,0), 2)
                # Add the bbox to the plot
                # Add label
        #### PUBLISH SEGMENTED IMAGE ####
        msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        msg.header.stamp = rospy.Time.now()
        self.image_pub.publish(msg)
         

    def load_image(self, image):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def segment_objects_on_white_image(self,image, boxes, masks, class_ids,
                                       scores=None,):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
        Returns result image.
        """
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        #xyz = rgb2xyz(image)
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]

        N = boxes.shape[0]

        white_image = np.zeros((height, width, channels), np.uint8)
        white_image[:] = (255, 255, 255)
        object_mask_image = np.zeros((height, width), np.uint8)
        kernel = np.ones((21,21),np.uint8)

        for i in range(N):

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue

            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            if(score < 0.5):
                break
            # Mask
            object_mask_image[:,:] = masks[:, :, i]
    
            contours, hierarchy = cv2.findContours(
                object_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.fillPoly(white_image, contours, (random.randint(
                0, 255), random.randint(0, 255), random.randint(0, 255)))
               
  
        #white_image = cv2.erode(white_image,kernel,iterations = 1)
        return white_image


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