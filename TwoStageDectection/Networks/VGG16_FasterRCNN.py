from __future__ import absolute_import
import jittor as jt
import jittor.nn as nn
from Networks.Parts.RPN import RPN
from Networks.Base_FasterRCNN import FasterRCNN
from Networks.Parts.Backbone_VGG import Vgg
from Networks.Parts.RoiPool import ROIPool

class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class=20,
                 ratios=None,
                 anchor_scales=None
                 ):

        if ratios is None:
            ratios = [0.5, 1, 2]

        if anchor_scales is None:
            anchor_scales = [8, 16, 32]

        info_cap, classifier = Vgg(setlist=[2, 2, 3, 3, 3])

        rpn = RPN(
            in_c=512, mid_c=512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            info_cap,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        jt.init.relu_invariant_gauss_(self.cls_loc.weight, mode="fan_out")
        jt.init.relu_invariant_gauss_(self.score.weight, mode="fan_out")

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = ROIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def execute(self, x, rois, roi_indices):
        roi_indices = jt.array(roi_indices).float()
        indices_and_rois = jt.concat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi.execute(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


