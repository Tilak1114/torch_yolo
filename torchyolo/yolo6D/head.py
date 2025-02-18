import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils.tal import make_anchors, dist2bbox
from torch.nn import functional as F
from ultralytics.nn.modules.block import DFL
from pipelearn.models.yolo6D.utils import rotation_decode, translation_decode


class Pose6DHead(nn.Module):
    """Detection head for rotation, translation, and classification."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers

        self.reg_max = 16  # DFL channels
        self.no = (
            nc + self.reg_max * 4 + 6 + 2 + 1
        )  # Class + BBox + Rotation + Trans_xy + Trans_z
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(
            ch[0], min(self.nc, 100)
        )  # channels

        # BBox and Classification heads
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        # Rotation head
        self.rot_head = nn.ModuleList(
            nn.Sequential(
                DWConv(x, x, 3),
                Conv(x, x, 1),
                nn.Conv2d(x, 6, 1),  # Predicts 6D rotation representation
            )
            for x in ch
        )

        # Translation heads
        self.trans_xy_head = nn.ModuleList(
            nn.Sequential(
                DWConv(x, x, 3),
                Conv(x, x, 1),
                nn.Conv2d(x, 2, 1),
            )
            for x in ch
        )
        self.trans_z_head = nn.ModuleList(
            nn.Sequential(
                DWConv(x, x, 3),
                Conv(x, x, 1),
                nn.Conv2d(x, 1, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            # Concatenate bbox, classification, rotation, translation predictions
            x[i] = torch.cat(
                (
                    self.cv2[i](x[i]),  # BBox predictions
                    self.cv3[i](x[i]),  # Classification predictions
                    self.rot_head[i](x[i]),  # Rotation predictions
                    self.trans_xy_head[i](x[i]),  # Trans_xy predictions
                    self.trans_z_head[i](x[i]),  # Trans_z predictions
                ),
                1,
            )
        if self.training:
            return x
        y = self._inference(x)
        return y, x

    def _inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        # Split concatenated tensor into components
        box_preds, cls_preds, rot_preds, trans_preds = x_cat.split(
            [self.reg_max * 4, self.nc, 6, 3], dim=1
        )

        # Decode bboxes
        dbox = (
            self.decode_bboxes(self.dfl(box_preds), self.anchors.unsqueeze(0))
            * self.strides
        )

        # Decode rotation using same method as loss.py
        rot_matrices = rotation_decode(rot_preds.permute(0, 2, 1))

        # Decode translation using same method as loss.py
        translations = translation_decode(
            trans_preds.permute(0, 2, 1),  # (B, N, 3)
            self.anchors.unsqueeze(0).permute(0, 2, 1),     # (1, N, 2)
            self.strides.unsqueeze(0).permute(0, 2, 1),                  # (N,)
        )

        # Apply sigmoid activation to classification logits
        cls_preds = cls_preds.sigmoid()

        # Reshape rotation matrices to match expected output format
        b, n, _, _ = rot_matrices.shape
        rot_reshaped = rot_matrices.permute(0, 2, 3, 1).reshape(b, 9, n)
        trans_reshaped = translations.transpose(1, 2)

        return torch.cat((dbox, cls_preds, rot_reshaped, trans_reshaped), dim=1)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    def bias_init(self):
        """Initialize biases for each head."""
        m = self
        for a, b, rot, trans_xy, trans_z, s in zip(
            m.cv2, m.cv3, m.rot_head, m.trans_xy_head, m.trans_z_head, m.stride
        ):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(
                5 / m.nc / (640 / s) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
            rot[-1].bias.data[:] = 0.0  # Initialize rotation biases to zero
            trans_xy[-1].bias.data[:] = 0.0  # Initialize Trans_xy biases to zero
            trans_z[-1].bias.data[:] = 0.0  # Initialize Trans_z biases to zero
