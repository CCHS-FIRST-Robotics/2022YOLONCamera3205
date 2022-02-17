import numpy as np
import torch
import os
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import deeppruner
import torchvision.transforms as transforms
from PIL import Image
import argparse
from models.config import config as config_args
import cv2


class DPNN:
    def __init__(self, CAM_DIST, DISP_CAL, PRUNED_PC_SIZE):
        self.PRUNED_PC_SIZE = PRUNED_PC_SIZE

        self.CAM_DIST = CAM_DIST

        self.DISP_CAL = DISP_CAL

        parser = argparse.ArgumentParser(description='DeepPruner')
        args = parser.parse_args()

        args.cost_aggregator_scale = config_args.cost_aggregator_scale
        args.downsample_scale = args.cost_aggregator_scale * 8.0
        args.cuda = True

        self.model = deeppruner.DeepPruner()
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        state_dict = torch.load("DeepPruner-fast-sceneflow.tar")
        self.model.load_state_dict(state_dict['state_dict'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def makeDisparityMap(self, l, r):
        limg_tensor = self.transform(l)
        rimg_tensor = self.transform(r)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()
        self.model.training = False
        with torch.no_grad():
            pred_disps = self.model(limg_tensor, rimg_tensor)
            pred_disps = pred_disps[0][:, :, :]
        predict_np = pred_disps.squeeze().cpu().numpy()
        return predict_np

    def makePC(self, l, r):
        disp_map = self.makeDisparityMap(l, r)
        img_points = cv2.reprojectImageTo3D(disp_map, self.DISP_CAL)
        img_points[:, :, 0] = img_points[:, :, 0] * -1
        img_points[:, :, 1] = img_points[:, :, 2] * -1
        img_points[:, :, 2] = img_points[:, :, 1]

        points = img_points.copy()
        points.shape = (-1, 3)
        rand_list = np.random.randint(0, points.shape[0], size=[self.PRUNED_PC_SIZE])
        pruned_points = points[rand_list, :]
        return disp_map, pruned_points, img_points
