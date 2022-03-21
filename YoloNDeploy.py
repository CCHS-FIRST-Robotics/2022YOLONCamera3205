import torch
import numpy as np
import cv2
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

PIXELS = 224

class YoloNDeploy:
    def __init__(self):
        self.device = select_device()
        print(self.device)
        self.model = DetectMultiBackend("ytp.pt", device=self.device)
        #self.model.warmup(imgsz=(1, 3, 224, 224), half=False)  # warmup

    def img2tensor(self, l):
        init_size = l.shape
        l0 = cv2.resize(l, (PIXELS, PIXELS), interpolation=cv2.INTER_NEAREST)
        l0 = l0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        l0 = np.ascontiguousarray(l0)
        l0 = torch.tensor(l0)
        l0 = l0.to(self.device)
        l0 = l0.type(torch.cuda.FloatTensor)
        l0 = torch.reshape(l0, [1, 3, PIXELS, PIXELS])
        l0 = l0 / 255
        return l0

    def pred2box(self, pred, init_size):
        pred = non_max_suppression(pred, 0.7, 0.4, max_det=40)
        x_l = lambda x: round(((x - PIXELS / 2) * init_size[1] / PIXELS) + (init_size[1] / 2))
        y_l = lambda y: round(((y - PIXELS / 2) * init_size[0] / PIXELS) + (init_size[0] / 2))
        ball_list = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords((PIXELS, PIXELS), det[:, :4], (PIXELS, PIXELS, 3))
                for *xyxy, conf, cls in reversed(det):
                    # print(xyxy, conf.cpu().numpy())
                    b = xyxy
                    x1 = x_l(b[0].cpu().numpy())
                    y1 = y_l(b[1].cpu().numpy())
                    x2 = x_l(b[2].cpu().numpy())
                    y2 = y_l(b[3].cpu().numpy())
                    center = (round((x1 + x2) * 0.5), round((y1 + y2) * 0.5))
                    radius = max(abs(x1 - x2), abs(y1 - y2)) * 0.5
                    if cls.cpu().numpy() == 1:
                        # blue
                        col = "R"
                    else:
                        # red
                        col = "B"
                    ball_list += [[center[0], center[1], col, radius]]
        return ball_list

    def deploy(self, l, r):
        l0 = self.img2tensor(l)
        r0 = self.img2tensor(r)
        img = torch.cat([l0, r0], dim=0)
        pred = self.model(img, augment=False, visualize=False)
        lpred = pred[0:1, :, :]
        rpred = pred[1:2, :, :]
        lball = self.pred2box(lpred, l.shape)
        rball = self.pred2box(rpred, r.shape)
        return lball, rball