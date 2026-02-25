# Ultralytics YOLO 🚀, AGPL-3.0 license
# Refactored for Mamba-YOLO Object Detection only
"""Model head modules - Detection only."""

import math

import torch
import torch.nn as nn

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

from .block import DFL
from .conv import Conv
from .utils import bias_init_with_prob, linear_init

__all__ = ("Detect",)


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)
