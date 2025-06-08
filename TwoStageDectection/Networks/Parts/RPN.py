import jittor as jt
import jittor.nn as nn
import jittor.misc as misc

from Utils.Base.BBox import generate_anchor_base
from Utils.Base.Creator import ProposalCreator


class RPN(nn.Module):

    def __init__(
            self, in_c=512, mid_c=512, ratios=None,
            anchor_scales=None, feat_stride=16,
            proposal_creator_params=None,
    ):
        super().__init__()

        if proposal_creator_params is None:
            proposal_creator_params = dict()

        if anchor_scales is None:
            anchor_scales = [8, 16, 32]

        if ratios is None:
            ratios = [0.5, 1, 2]

        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=3, stride=1, padding=1)
        self.score = nn.Conv2d(mid_c, n_anchor * 2, kernel_size=1, stride=1, padding=0)
        self.loc = nn.Conv2d(mid_c, n_anchor * 4, kernel_size=1, stride=1, padding=0)

        jt.init.relu_invariant_gauss_(self.conv1.weight, mode="fan_out")
        jt.init.relu_invariant_gauss_(self.score.weight, mode="fan_out")
        jt.init.relu_invariant_gauss_(self.loc.weight, mode="fan_out")

    def execute(self, x, img_size, scale=1.):

        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            self.anchor_base,
            self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        x = self.conv1(x)
        x = nn.relu(x)

        rpn_locs = self.loc(x)

        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(x)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()

        rpn_softmax_scores = nn.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()

        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i],
                rpn_fg_scores[i],
                anchor, img_size,
                scale=scale)

            batch_index = i * jt.ones((len(roi),), dtype='int32')
            rois.append(roi)

            roi_indices.append(batch_index)

        roi_out = jt.concat(rois, dim=0)
        roi_indices = jt.concat(roi_indices, dim=0)
        return rpn_locs, rpn_scores, roi_out, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):

    shift_y = jt.arange(0, height * feat_stride, feat_stride)
    shift_x = jt.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = jt.meshgrid(shift_x, shift_y)
    shift = jt.stack((jt.flatten(shift_y), jt.flatten(shift_x),
                      jt.flatten(shift_y), jt.flatten(shift_x)), dim=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype('float32')
    return anchor
