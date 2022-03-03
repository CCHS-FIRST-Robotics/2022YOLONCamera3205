import torch
import torch.backends.cudnn as cudnn
import cv2

from yolox.core import launch
from yolox.exp import get_exp
from yolox.data import ValTransform
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger, postprocess


# -n  yolox-s -c {MODEL_PATH} -b 64 -d 1 --conf 0.001 -f

class YoloxDeploy:
    def __init__(self):
        self.rank = 0
        self.exp = get_exp(None, "yolox-s")
        self.model = self.exp.get_model()
        torch.cuda.set_device(self.rank)
        self.model.cuda(self.rank)
        self.model.eval()
        loc = "cuda:{}".format(self.rank)
        ckpt = torch.load("best_ckpt.pth.tar", map_location=loc)
        self.model.load_state_dict(ckpt["model"])
        self.preproc = ValTransform(
            rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def deploy(self, cv2img):
        init_size = cv2img.shape
        imgs = cv2.resize(cv2img, (416, 416), interpolation = cv2.INTER_AREA)
        imgs, _ = self.preproc(imgs, imgs, (416, 416))
        imgs = torch.tensor(imgs)
        imgs = torch.reshape(imgs, [1, 3, 416, 416])
        with torch.no_grad():
            imgs = imgs.type(torch.cuda.FloatTensor)
            outputs = self.model(imgs)
            outputs = postprocess(
                outputs, 2
            )
        x_l = lambda x: round(( (x - 416/2) * init_size[1]/416 ) + (init_size[1]/2))
        y_l = lambda y: round(( (y - 416/2) * init_size[0]/416 ) + (init_size[0]/2))
        ball_list = []
        if outputs[0] != None:
            b = outputs[0].cpu().detach().numpy()
            for c in range(outputs[0].size()[0]):
                x1 = x_l(b[c, 0])
                y1 = y_l(b[c, 1])
                x2 = x_l(b[c, 2])
                y2 = y_l(b[c, 3])
                c = round(b[c, 6])
                center = (round((x1+x2)*0.5), round((y1+y2)*0.5))
                radius = max(abs(x1 - x2), abs(y1 - y2)) * 0.4
                if c == 1:
                    #blue
                    col = "B"
                else:
                    #red
                    col = "R"
                ball_list += [[center[0], center[1], col, radius]]
        return ball_list