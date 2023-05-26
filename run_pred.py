from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
# import copy
# from matplotlib import pyplot as plt
# import pandas as pd
import os

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.utils.layer_utils import get_source_inputs
from keras.utils import layer_utils
from keras.models import Model
from keras.utils import generic_utils

from keras.objectives import categorical_crossentropy
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

from sklearn.metrics import average_precision_score

from vgg import *
from loss import *
from config import *

config_output_filename = 'model_vgg_config.pickle'
model_path = 'model\model_frcnn_vgg.hdf5'

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

        # define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

#print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

        # Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
# print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

bbox_threshold = 0.7

def describe_image(filepath):
    img = cv2.imread(filepath)
    X, ratio = format_img(img, C)
            
    X = np.transpose(X, (0, 2, 3, 1))

            # get output layer Y1, Y2 from the RPN and the feature maps F
            # Y1: y_rpn_cls
            # Y2: y_rpn_regr
    [Y1, Y2, F] = model_rpn.predict(X)

            # Get bboxes by applying NMS 
            # R.shape = (300, 4)
    R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0]//C.num_rois:
                    #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
                    # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
    if len(bboxes) == 0 or len(probs) == 0:
        return "[{'topwear_type': 'Not detected', 'conf': 0},{'sleeve_type': 'Not detected', 'conf': 0},{'neck_type': 'Not detected', 'conf': 0},{'design_type': 'Not detected', 'conf': 0},{'bottomwear_type': 'Not detected', 'conf': 0},{'footwear_type': 'Not detected', 'conf': 0}]"

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

                        # Calculate real coordinates on original image
            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            #cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)
            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            #(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            #textOrg = (real_x1, real_y1-0)

            # cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
            # cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            # cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            sort_dets = sorted(all_dets)

            dets_dict = {}
            for name, score in sort_dets:
                dets_dict[name] = score

                        #list of apparel attributres

            top_wear_lst = ['tshirt','shirt','dress','jacket','suit', 'sweatshirt','tanktop','jacket','rainjacket','leatherjacket','denimjacket','vest','top','shrug','jumpsuit','pulloversweater','coat']
            sleeve_lst = ['fullsleeve', 'halfsleeve','sleeveless']
            neck_lst = ['turtleneck','vneck','roundneck','squareneck','tabcollar','mandarin','poloneck']
            design_lst = ['printed','pattern','floral','dotted','solid','striped']
            footwear_lst = ['shoes','boots','sandals','slippers','sneakers','clogs']
            bottomwear_lst = ['pants','jeans','shorts','leggings','sweatpants','brief','trackpants','trunk']

            topwear_dict= {}
            sleeve_dict = {}
            design_dict = {}
            neck_dict = {}
            bottomwear_dict = {}
            footwear_dict = {}

                        #default no detection
                        #topwear
            topwear_dict["topwear_type"] = 'Not detected'
            topwear_dict["conf"] = 0
                        #sleeve
            sleeve_dict["sleeve_type"] = 'Not detected'
            sleeve_dict["conf"] = 0
                        #design
            design_dict["design_type"] = 'Not detected'
            design_dict["conf"] = 0
                        #neck
            neck_dict["neck_type"] = 'Not detected'
            neck_dict["conf"] = 0
                        #bottomwear
            bottomwear_dict["bottomwear_type"] = 'Not detected'
            bottomwear_dict["conf"] = 0
                        #footwear
            footwear_dict["footwear_type"] = 'Not detected'
            footwear_dict["conf"] = 0

            final_dets = []

            final_dets.append(topwear_dict)
            final_dets.append(sleeve_dict)
            final_dets.append(neck_dict)
            final_dets.append(design_dict)
            final_dets.append(bottomwear_dict)
            final_dets.append(footwear_dict)


            for key, value in dets_dict.items():
                if key in top_wear_lst:
                    topwear_dict["topwear_type"] = key
                    topwear_dict["conf"] = value  
                                
            for key, value in dets_dict.items():
                if key in sleeve_lst:
                    sleeve_dict["sleeve_type"] = key
                    sleeve_dict["conf"] = value

            for key, value in dets_dict.items():
                if key in neck_lst:
                    neck_dict["neck_type"] = key
                    neck_dict["conf"] = value

            for key, value in dets_dict.items():
                if key in design_lst:
                    design_dict["design_type"] = key
                    design_dict["conf"] = value

            for key, value in dets_dict.items():
                if key in bottomwear_lst:
                    bottomwear_dict["bottomwear_type"] = key
                    bottomwear_dict["conf"] = value

            for key, value in dets_dict.items():
                if key in footwear_lst:
                    footwear_dict["footwear_type"] = key
                    footwear_dict["conf"] = value
    return final_dets 