from __future__ import absolute_import
from __future__ import division
import jittor as jt
import jittor.nn as nn
from Utils.Base.BBox import bbox_decode
from jittor.misc import nms

# from model.utils.nms import non_maximum_suppression

def nograd(f):
    def new_f(*args, **kwargs):
        with jt.no_grad():
            return f(*args, **kwargs)

    return new_f


class FasterRCNN(nn.Module):

    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=None,
                 loc_normalize_std=None
                 ):
        super(FasterRCNN, self).__init__()

        if loc_normalize_mean is None:
            loc_normalize_mean = (0., 0., 0., 0.),

        if loc_normalize_std is None:
            loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def execute(self, x, scale=1.):

        img_size = x.shape[2:]

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(
            h, rois, roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

    def use_preset(self, preset):

        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l, self.nms_thresh)

            bbox.append(cls_bbox_l[keep].cpu().numpy())

            label.append((l - 1) * jt.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        bbox = jt.concat(bbox, axis=0).astype(jt.float32)
        label = jt.concat(label, axis=0).astype(jt.int32)
        score = jt.concat(score, axis=0).astype(jt.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs, sizes=None, visualize=False):

        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = jt.array(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale)
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = jt.array(rois) / scale

            mean = jt.array(self.loc_normalize_mean). \
                repeat(self.n_class)[None]
            std = jt.array(self.loc_normalize_std). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = bbox_decode(roi.reshape((-1, 4)),
                                   roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (nn.softmax(roi_score, dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        return bboxes, labels,