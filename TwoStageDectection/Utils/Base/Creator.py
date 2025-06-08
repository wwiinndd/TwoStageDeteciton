import jittor as jt
from jittor.misc import nms
import jittor.init as jtinit
import jittor.misc as misc
from Utils.Base.BBox import bbox_decode, bbox_encode, bbox_iou


class ProposalTargetCreator(object):

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.1
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):

        n_bbox, _ = bbox.shape

        roi = jt.concat([roi, bbox], dim=0)

        pos_roi_per_image = jt.round(jt.array(self.n_sample * self.pos_ratio))
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(dim=1)[0]
        max_iou = iou.max(dim=1)

        gt_roi_label = label[gt_assignment] + 1

        pos_index = jt.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image[0], jt.get_len(pos_index)))

        if jt.get_len(pos_index) > 0:
            rand_idx = jt.randperm(jt.get_len(pos_index))
            pos_index = pos_index[rand_idx[:pos_roi_per_this_image]]

        neg_index = jt.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, jt.get_len(neg_index)))

        if jt.get_len(neg_index) > 0:
            rand_idx = jt.randperm(jt.get_len(neg_index))
            neg_index = neg_index[rand_idx[:neg_roi_per_this_image]]


        keep_index = jt.concat([pos_index, neg_index], dim=0)

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]



        gt_roi_loc = bbox_encode(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - jt.array(loc_normalize_mean, dtype="float32")
                       ) / jt.array(loc_normalize_std,  dtype="float32"))

        return sample_roi, gt_roi_loc, gt_roi_label


class AnchorTargetCreator(object):

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):

        img_H, img_W = img_size

        n_anchor = jt.get_len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        loc = bbox_encode(anchor, bbox[argmax_ious])

        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        label = jt.empty((len(inside_index),), dtype='int32')
        jtinit.fill(label, -1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0

        label[gt_argmax_ious] = 1

        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = jt.where(label == 1)[0]

        if jt.get_len(pos_index) > n_pos:
            rand_index = jt.randperm(len(pos_index) - n_pos)
            disable_index = pos_index[rand_index]
            label[disable_index] = -1

        n_neg = self.n_sample - jt.sum(label == 1)
        neg_index = jt.where(label == 0)[0]

        if jt.get_len(neg_index) > n_neg:
            rand_index = jt.randperm(len(neg_index) - n_neg.item())
            disable_index = neg_index[rand_index]
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(dim=1)[0]
        max_ious = ious[jt.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(dim=0)[0]
        gt_max_ious = ious[gt_argmax_ious, jt.arange(ious.shape[1])]
        gt_argmax_ious = jt.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):

    if len(data.shape) == 1:
        ret = jt.empty((count,), dtype=data.dtype)
        jtinit.fill(ret, fill)
        ret[index] = data
    else:
        ret = jt.empty((count,) + data.shape[1:], dtype=data.dtype)
        jtinit.fill(ret, fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):

    index_inside = jt.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


class ProposalCreator:

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = bbox_decode(anchor, loc)

        roi[:, 0:4:2] = jt.safe_clip(
            roi[:, 0:4:2], 0, img_size[0])
        roi[:, 1:4:2] = jt.safe_clip(
            roi[:, 1:4:2], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]


        keep = jt.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = misc.sort(jt.flatten(score), descending=True)[0]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        score = jt.unsqueeze(score, dim=1)
        rois = jt.concat([roi, score], dim=1)

        keep = nms(
            rois,
            self.nms_thresh)

        rois = rois[keep]
        roi = rois[:,0:4]

        return roi