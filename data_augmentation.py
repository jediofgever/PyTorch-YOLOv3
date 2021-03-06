import json
import cv2 
import numpy as np

import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


## Open lables which is in coco Format see here for this format http://cocodataset.org/#format-data 
with open('data/custom/images/labels.json') as f:
  data =  json.load(f)

#parse this jsoin and get annotion and image seperately
annots = data['annotations']
images = data['images']

##Open train.txt and valid.txt to write the path of training and validation images
file_that_contains_training_image_paths = open("data/custom/train.txt","w+") 
file_that_contains_validation_image_paths = open("data/custom/valid.txt","w+") 

#define sequnce for ramdimizing data
# we use https://github.com/aleju/imgaug, all images will be go through this sequance and data augmentation willbe achieved

 
# Define our sequence of augmentation steps that will be applied to every image.
static_seq = iaa.Sequential([
      iaa.Multiply((0.8, 1.3)), # change brightness, doesn't affect BBs
      iaa.Affine(
          translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
          scale=(0.6, 0.9)
      ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])
 
# Define our sequence of augmentation steps that will be applied to every image.
hard_seq = iaa.Sequential([
      iaa.Multiply((0.8, 1.2)), # change brightness, doesn't affect BBs
      iaa.Affine(
          translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
          scale=(0.7, 1.2),
      ),
      iaa.AdditiveGaussianNoise(scale=0.05*255),
      iaa.Crop(px=(1, 16), keep_size=False),
      iaa.Fliplr(0.5),
])  


def write_augmented_image_and_labels(image_aug , bbs_aug,class_id, prefix,image_file_name, label_file_name):

  file_that_contains_validation_image_paths.write('data/custom/images/'+prefix+ image_file_name +'\n')
  file_that_contains_training_image_paths.write('data/custom/images/'+prefix+image_file_name+'\n')

  augmented_label = open("data/custom/labels/"+prefix+label_file_name,"w+")
  

  for i in range(len(bbs_aug)):
 
    x1 = int(bbs_aug[i].x1)
    x2 = int(bbs_aug[i].x2)

    y1 = int(bbs_aug[i].y1)
    y2 = int(bbs_aug[i].y2)
     
    if x1 < 0:
      x1=1
    if x1 > 960:
      x1 = 959
    if x2 < 0:
      x2=1
    if x2 > 960:
      x2 = 959
    if y1 < 0:
      y1=1
    if y1 > 960:
      y1 = 959
    if y2 < 0:
      y2=1
    if y2 > 960:
      y2 = 959
    print(x1,y1)
    #cv2.rectangle(image_aug,(x1,y1),(x2,y2),(90,0,0),1)

    center_x_augmented = (x1 + x2)/2 
    center_y_augmented = (y1 + y2)/2 
    width_augmented = (x2 - x1) 
    height_augmented = (y2 - y1) 

    max_ = 960
    min_ = 0 

    normalized_center_x = center_x / max_
    normalized_center_y = center_y / max_
    normalized_width = width / max_
    normalized_height = height / max_

    augmented_label.write(str(class_id) + ' '+ 
                          str(normalized_center_x) + ' '+  
                          str(normalized_center_y) + ' '+ 
                          str(normalized_width) + ' '+ 
                          str(normalized_height) + '\n')
  cv2.imwrite("data/custom/images/"+prefix+str(images[k]['file_name']),image_aug)
  augmented_label.close()

                       
## Go through all images that are labeled  
for k in range (len(images)):

  ## Write Pathof original Images
  file_that_contains_training_image_paths.write('data/custom/images/'+str(images[k]['file_name'])+'\n')
  
  ## Write Path of validation Images
  file_that_contains_validation_image_paths.write('data/custom/images/'+str(images[k]['file_name'])+'\n')
  
  ##Open a file for each image thats has labeled objectsin
  file_that_contains_labels_for_this_image = open("data/custom/labels/"+str(images[k]['file_name'][:-4])+".txt","w+")
  ## a array to store the boxes for that define boundries of labels
  
  #Get original image, that is labeled
  original_image = cv2.imread("data/custom/images/"+str(images[k]['file_name']))
 
  ## The very first images are 1280x720 , so we need to resize ONCE them and ALWAYS change labels accordingly
  if(original_image.shape[0] != original_image.shape[1]):
    original_image = cv2.copyMakeBorder(original_image, 0, 560, 0, 0, cv2.BORDER_CONSTANT) 
  original_image = cv2.resize(original_image, (960,960), interpolation = cv2.INTER_AREA)



  all_boxes_of_this_image=[]
  
  num_annots_in_this_image = 0

  ##And now go through all the annotations and match them with the images they are in
  for i in range(len(annots)):

    ## This BBX for mat is top_left_x, top_left_y, width, height
    bounding_box_of_this_annotation = annots[i]['bbox'] 
       
    ## But YOLO format is; 
    ## class_id center_x center_y width height
    ## lets do conversion to format of center_x center_y width height, MULTIPLY THEM BY 0.5 BECAUSE WE DOWNSIZED THE ORIGINAL IMAGE TO 960X360
    ## SO LABELS SHOULD BE DOWNSIZED BY HALF TOO
    center_x = (bounding_box_of_this_annotation[0]+ (bounding_box_of_this_annotation[2]/2)) *960/1280
    center_y = (bounding_box_of_this_annotation[1]+ (bounding_box_of_this_annotation[3]/2)) *960/1280
    width = (bounding_box_of_this_annotation[2]) * 960/1280
    height =(bounding_box_of_this_annotation[3]) * 960/1280
    

    
    
    max_ = original_image.shape[1] 
    min_ = 0 
    ## This data should be normalized BETWEEN [0,1]
    normalized_center_x = center_x / max_
    normalized_center_y = center_y / max_
    normalized_width = width / max_
    normalized_height = height / max_
    
    ##GET THE image_id that contains this bounding box
    image_id_that_has_this_bounding_box = int(annots[i]['image_id'])
    

    # Now lets match this bounding box with its parent image 
    if(images[k]['id'] == image_id_that_has_this_bounding_box):
      print(bounding_box_of_this_annotation)
      num_annots_in_this_image += 1

      ## WRITE THIS ANNOTATION WITH MATCHED BOUNDING BOX
      file_that_contains_labels_for_this_image.write(str(0) + ' '+ 
                                  str(normalized_center_x) + ' '+  
                                  str(normalized_center_y) + ' '+ 
                                  str(normalized_width) + ' '+ 
                                  str(normalized_height) + '\n')
      
      ##
      this_bounding_box = [BoundingBox(x1=int(center_x-width/2), x2=int(center_x+width/2), y1=int(center_y-height/2), y2=int(center_y+height/2))]
      #cv2.rectangle(original_image, (int(center_x-width/2), int(center_y-height/2)), (int(center_x+width/2), int(center_y+height/2)), (255,0,0), 2)

      ## append this box
      all_boxes_of_this_image.extend(this_bounding_box)
  if num_annots_in_this_image == 0:
        continue

  image_file_name = images[k]['file_name']
  cv2.imwrite("data/custom/images/"+image_file_name,original_image)

  label_file_name = images[k]['file_name'][:-4]+ ".txt"

  original_bounding_boxes = BoundingBoxesOnImage(all_boxes_of_this_image,shape=original_image.shape)


  prefix = 'augmentation_one_'
  images_augmentation_one_, bbs_augmentation_one_ = hard_seq(image=original_image, bounding_boxes=original_bounding_boxes)
  bbs_augmentation_one_.remove_out_of_image()
  bbs_augmentation_one_.remove_out_of_image().clip_out_of_image()
  write_augmented_image_and_labels(images_augmentation_one_,bbs_augmentation_one_,0,prefix,image_file_name,label_file_name)

  prefix = 'augmentation_second_'
  images_augmentation_second_, bbs_augmentation_second_ = static_seq(image=original_image, bounding_boxes=original_bounding_boxes)
  bbs_augmentation_second_.remove_out_of_image()
  bbs_augmentation_second_.remove_out_of_image().clip_out_of_image()
  write_augmented_image_and_labels(images_augmentation_second_,bbs_augmentation_second_,0,prefix,image_file_name,label_file_name)
  

    


  '''
  
  prefix = 'augmentation_second_'
  images_augmentation_second_, bbs_augmentation_second_ = static_seq(image=original_image, bounding_boxes=original_bounding_boxes)
  bbs_augmentation_second_.remove_out_of_image()
  bbs_augmentation_second_.remove_out_of_image().clip_out_of_image()
  write_augmented_image_and_labels(images_augmentation_second_,bbs_augmentation_second_,0,prefix,image_file_name,label_file_name)
  
  prefix = 'augmentation_second_'
  images_augmentation_second_, bbs_augmentation_second_ = seq(image=original_image, bounding_boxes=original_bounding_boxes)
  write_augmented_image_and_labels(images_augmentation_second_,bbs_augmentation_second_,annots[i]['category_id'],prefix,image_file_name,label_file_name)
    
  prefix = 'augmentation_third_'
  images_augmentation_third_, bbs_augmentation_third_ = seq(image=original_image, bounding_boxes=original_bounding_boxes)
  write_augmented_image_and_labels(images_augmentation_third_,bbs_augmentation_third_,annots[i]['category_id'],prefix,image_file_name,label_file_name)
   
  prefix = 'augmentation_forth_'
  images_augmentation_forth_, bbs_augmentation_forth_ = seq(image=original_image, bounding_boxes=original_bounding_boxes)
  write_augmented_image_and_labels(images_augmentation_forth_,bbs_augmentation_forth_,annots[i]['category_id'],prefix,image_file_name,label_file_name) 
  ''' 
file_that_contains_training_image_paths.close()
file_that_contains_validation_image_paths.close()