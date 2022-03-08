import torch
import numpy as np
import cv2
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

class YoloNDeploy:
    def __init__(self):
        self.device = select_device()
        print(self.device)
        self.model = DetectMultiBackend("best5.pt", device=self.device, dnn = False, data = "data/coco128.yaml")
        self.model.warmup(imgsz=(1, 3, 416, 416), half=False)  # warmup

    def deploy(self, cv2img):
        init_size = cv2img.shape
        imgs = cv2.resize(cv2img, (416, 416), interpolation = cv2.INTER_AREA)
        imgs = imgs.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        imgs = np.ascontiguousarray(imgs)
        imgs = torch.tensor(imgs)
        imgs = imgs.to(self.device)
        imgs = imgs.type(torch.cuda.FloatTensor)
        imgs = torch.reshape(imgs, [1, 3, 416, 416])
        pred = self.model(imgs/255, augment = False, visualize = False)
        pred = non_max_suppression(pred, 0.8, 0.45, max_det = 4)
        x_l = lambda x: round(( (x - 416/2) * init_size[1]/416 ) + (init_size[1]/2))
        y_l = lambda y: round(( (y - 416/2) * init_size[0]/416 ) + (init_size[0]/2))
        ball_list = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((416,416), det[:, :4], (416,416,3))
                for *xyxy, conf, cls in reversed(det):
                    #print(xyxy, conf.cpu().numpy())
                    b = xyxy
                    x1 = x_l(b[0].cpu().numpy())
                    y1 = y_l(b[1].cpu().numpy())
                    x2 = x_l(b[2].cpu().numpy())
                    y2 = y_l(b[3].cpu().numpy())
                    center = (round((x1 + x2) * 0.5), round((y1 + y2) * 0.5))
                    radius = max(abs(x1 - x2), abs(y1 - y2)) * 0.4
                    if cls.cpu().numpy() == 1:
                        # blue
                        col = "R"
                    else:
                        # red
                        col = "B"
                    ball_list += [[center[0], center[1], col, radius]]
        return ball_list