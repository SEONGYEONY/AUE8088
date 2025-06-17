import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

class HeadInfo(nn.Module):
    """
    Dummy head wrapper to expose attributes needed by ComputeLoss (na, nc, nl, anchors, stride).
    """
    def __init__(self, na: int, nc: int, anchors: torch.Tensor, stride: list[int]):
        super().__init__()
        self.na = na            # number of anchors
        self.nc = nc            # number of classes
        self.nl = 1             # number of detection layers
        self.stride = stride    # list of stride(s)
        # register anchors as a buffer so they're moved to the correct device
        self.register_buffer('anchor_tensor', anchors)
        # Detect code expects a list of length nl
        self.anchors = [self.anchor_tensor]

    def forward(self, x):
        # unused in loss; detection logic in MultiStreamCNN
        return x

class MultiStreamCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        num_anchors: int = 3,
        hyp: dict | None = None,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
    ):
        super().__init__()
        # --- hyperparams for ComputeLoss ---
        self.hyp = hyp or {
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'label_smoothing': 0.0,
            'fl_gamma': 0.0,
            'iou_t': 0.2,
            'anchor_t': 4.0,
            'box': 1.0,
            'obj': 1.0,
            'cls': 1.0,
        }
        self.nc = num_classes
        self.na = num_anchors
        self.nl = 1

        # --- feature extractor streams ---
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.ir_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # fuse and optionally pool (you can remove AdaptiveAvgPool2d to keep spatial dims)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # detection head: predict (5 + nc) per anchor
        out_channels = self.na * (5 + self.nc)
        self.det_head = nn.Conv2d(128, out_channels, kernel_size=1)

        # dummy anchors (width, height) -- set roughly to 1x1 cells
        anchor_tensor = torch.ones(self.na, 2)
        self.register_buffer('anchor_tensor', anchor_tensor)
        self.anchors = [self.anchor_tensor]

        # downsample factor: two 2× pools => stride 4
        self.stride = [4]

        # metadata wrapper for ComputeLoss: model.model[-1] must be HeadInfo
        self.model = nn.ModuleList([HeadInfo(self.na, self.nc, self.anchor_tensor, self.stride)])

        # thresholds for inference
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def forward(self, x):
        # 1) extract
        f1 = self.rgb_stream(x)
        f2 = self.ir_stream(x)
        # 2) fuse
        f = torch.cat([f1, f2], dim=1)
        f = self.fuse_conv(f)
        # 3) predict raw preds
        p = self.det_head(f)
        B, C, H, W = p.shape
        # reshape to (B, A, H, W, 5+nc)
        p = p.view(B, self.na, 5 + self.nc, H, W).permute(0, 1, 3, 4, 2).contiguous()
        train_out = [p]

        # 3) post-process for preds
        # split coords + objectness
        boxes  = p[..., :4]               # (B,A,H,W,4)
        logits = p[..., 4]                # (B,A,H,W)
        scores = torch.sigmoid(logits)    # (B,A,H,W)

        # flatten
        boxes  = boxes.view(B, -1, 4)     # (B, A*H*W, 4)
        scores = scores.view(B, -1)       # (B, A*H*W)

        preds = []
        for b in range(B):
            keep = scores[b] > self.conf_thresh
            if keep.sum() == 0:
                preds.append(torch.zeros((0,4), device=boxes.device))
                continue
            b_boxes  = boxes[b][keep]      # (M,4)
            b_scores = scores[b][keep]     # (M,)
            keep2    = nms(b_boxes, b_scores, self.iou_thresh)
            final    = b_boxes[keep2]      # (Ni,4)
            preds.append(final)

        # return two values so that
        #   preds, train_out = model(ims)
        # works correctly
        return preds, train_out

    def detect(self, x):
        """
        Inference method: apply sigmoid on objectness, NMS, etc.
        Returns lists of x1, x2, y1, y2 per batch.
        """
        p_list = self.forward(x, x)[0]  # (B,A,H,W,5+nc)
        B, A, H, W, _ = p_list.shape
        # objectness logits at index 4
        logits = p_list[..., 4]
        boxes = p_list[..., :4]
        scores = torch.sigmoid(logits)
        boxes = boxes.view(B, -1, 4)
        scores = scores.view(B, -1)

        all_x1, all_y1, all_x2, all_y2 = [], [], [], []
        for b in range(B):
            keep = scores[b] > self.conf_thresh
            if keep.sum() == 0:
                all_x1.append(torch.empty(0))
                all_y1.append(torch.empty(0))
                all_x2.append(torch.empty(0))
                all_y2.append(torch.empty(0))
                continue

            b_boxes = boxes[b][keep]
            b_scores = scores[b][keep]
            keep2 = nms(b_boxes, b_scores, self.iou_thresh)
            final = b_boxes[keep2]
            x1, y1, x2, y2 = final.unbind(1)
            all_x1.append(x1)
            all_y1.append(y1)
            all_x2.append(x2)
            all_y2.append(y2)

        return all_x1, all_x2, all_y1, all_y2
