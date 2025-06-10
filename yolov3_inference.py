import os
import shutil
import numpy as np
import cv2
import copy
import time
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from yolov3.detect_infrared import load_infrared_model, detect_infrared
from yolov3.detect_visible import load_visible_model, detect_visible
from itertools import chain
from DE import DifferentialEvolutionAlgorithm

from utils.draw_utils import draw_predictions, draw_all_predictions

content = 1

trans = transforms.Compose([
                transforms.ToTensor(),
            ])

threat_infrared_model = load_infrared_model()

# dataset_name = 'llvip_dataset_50ims'
dataset_name = 'llvip_dataset_1im_atk'
infrared_dir = f'{dataset_name}/infrared'

def limit_region(bbox):
    x_left = bbox[0] + (bbox[2] - bbox[0]) / 4
    x_right = bbox[2] - (bbox[2] - bbox[0]) / 4
    y_low = bbox[1]
    y_high = bbox[3]
    y_head = y_low + (y_high - y_low) / 4
    y_leg = y_low + (y_high - y_low) / 2
    return x_left, x_right, y_head, y_leg

def get_state(img_path, bbox):
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    points = []
    patch_1 = []
    patch_2 = []
    w_step = int(bbox_width / 10)
    h_step = int(bbox_height / 10)
    bbox = list(map(int, bbox))
    x_left, x_right = bbox[0], bbox[2]
    y_up, y_below = bbox[1], bbox[3]

    px_1 = (y_up+2.5*h_step + y_up+3*h_step) / 2
    py_1 = (x_left+5*w_step + x_right-3*w_step) / 2

    px_2 = (y_up+4*h_step + y_up+6*h_step) / 2
    py_2 = (x_right-5*w_step + x_right-4*w_step) / 2

    a = 12 
    e = 15
    eq_points = []
    state = []
    # ---patch 1
    eq_points_1 = []
    points_1 = []
    for n in range(1,a+1):
        xx = px_1 + round(e*np.cos(2*np.pi*(n-1)/a),2)  #
        yy = py_1 + round(e*np.sin(2*np.pi*(n-1)/a),2)
        eq_points_1.append([xx,yy]) 
    eq_points_1.append([px_1 + round(e*np.cos(0),2), py_1 + round(e*np.sin(0),2)])
    for i in range(len(eq_points_1)-1):
        pre_x = eq_points_1[i][0]
        pre_y = eq_points_1[i][1]
        x = eq_points_1[i+1][0]
        y = eq_points_1[i+1][1]
        points_1.append([int(round((pre_x+x)/2,2)),int(round((pre_y+y)/2,2))])

    # ---patch 2
    eq_points_2 = []
    points_2 = []
    for n in range(1,a+1):
        xx = px_2 + round(e*np.cos(2*np.pi*(n-1)/a),2)
        yy = py_2 + round(e*np.sin(2*np.pi*(n-1)/a),2)
        eq_points_2.append([xx,yy]) 
    eq_points_2.append([px_2 + round(e*np.cos(0),2), py_2 + round(e*np.sin(0),2)])
    for i in range(len(eq_points_2)-1):
        pre_x = eq_points_2[i][0]
        pre_y = eq_points_2[i][1]
        x = eq_points_2[i+1][0]
        y = eq_points_2[i+1][1]
        points_2.append([int(round((pre_x+x)/2,2)),int(round((pre_y+y)/2,2))])

    eq_points.append(eq_points_1)
    eq_points.append(eq_points_2)
    state.append(points_1)
    state.append(points_2)

    return px_1, py_1, px_2, py_2, eq_points, state

if __name__ == "__main__":
    for img_path in tqdm(os.listdir(infrared_dir)):
        infrared_img = infrared_dir + '/' + img_path
        infrared_sample = Image.open(infrared_img)
        infrared_input = trans(infrared_sample) # to tensor
        infrared_ori = torch.stack([infrared_input]) # N C H W
        infrared_det = F.interpolate(infrared_ori, (416, 416), mode='bilinear', align_corners=False) # 采用双线性插值将不同大小图片上/下采样到统一大小
        H, W = infrared_sample.size[1], infrared_sample.size[0]
        bboxes, probs_infrared = detect_infrared(threat_infrared_model, infrared_det)
        bbox = bboxes[0]
        prob_infrared = probs_infrared[0]
        bbox[0], bbox[1], bbox[2], bbox[3] = int(bbox[0]*W/416), int(bbox[1]*H/416), int(bbox[2]*W/416), int(bbox[3]*H/416)
        print(img_path)
        print('Origin infared score: {}'.format(prob_infrared))

        out_inf_dir = f'inference_results/{dataset_name}/orig_predictions/infrared'
        os.makedirs(out_inf_dir, exist_ok=True)
        draw_predictions(infrared_ori, bbox, prob_infrared, os.path.join(out_inf_dir, f"pred_{img_path}"))
        draw_all_predictions(infrared_ori, bboxes, probs_infrared, os.path.join(out_inf_dir, f"pred_{img_path}"))

