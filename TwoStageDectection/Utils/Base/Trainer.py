from __future__ import absolute_import
import os
from collections import namedtuple
import time

from Utils.Base.Losses import _fast_rcnn_loc_loss
from Utils.Base.Creator import AnchorTargetCreator, ProposalTargetCreator
import jittor as jt
import jittor.nn as nn

# from utils.vis_tool import Visualizer

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):

    def __init__(self, faster_rcnn, rpn_sigma, roi_sigma, optimizer, scheduler):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = optimizer
        self.scheduler = scheduler

    def execute(self, imgs, bboxes, labels, scale=1.):
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]


        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            rois,
            bbox,
            label,
            self.loc_normalize_mean,
            self.loc_normalize_std)

        sample_roi_index = jt.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox,
            anchor,
            img_size)
        gt_rpn_label = gt_rpn_label.long()
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        rpn_cls_loss = nn.cross_entropy_loss(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[jt.arange(0, n_sample).long(), gt_roi_label.long()]
        gt_roi_label = gt_roi_label.long()

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.detach(),
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale=1.):
        self.optimizer.zero_grad()
        losses = self.execute(imgs, bboxes, labels, scale)
        self.optimizer.step(losses.total_loss)
        self.scheduler.step()
        return losses
