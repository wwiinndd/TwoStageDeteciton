from __future__ import division

from collections import defaultdict
import itertools
import jittor as jt
from Utils.Base.BBox import bbox_iou

def nanmean(inputs):
    CUDA_HEADER = r'''
    #include <cmath>
    using namespace std;
    '''

    CUDA_SRC = r"""
        __global__ static void kernel1(@ARGS_DEF) {
            @PRECALC
            float sum = 0;
            int ing_sum = 0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < in0_shape0; i += blockDim.x * gridDim.x){
                __isnanf(@in0(i)) ? ing_sum += 1 : sum += @in0(i);
            }
            int all_len = in0_shape0 - ing_sum;
            all_len == 0 ? @out(0) = NAN : @out(0) = sum / all_len;
        }
        const int total_count = in0_shape0;
        const int thread_per_block = 1024;
        const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
        kernel1<<<block_count, thread_per_block>>>(@ARGS);
    """

    return jt.code(shape=(1,), dtype='float32', inputs=[inputs], cuda_header=CUDA_HEADER, cuda_src=CUDA_SRC)


def nan_to_num(input):
    CUDA_HEADER = r'''
    #include <cmath>
    using namespace std;
    '''

    CUDA_SRC = r"""
        __global__ static void kernel1(@ARGS_DEF) {
            @PRECALC
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < in0_shape0; i += blockDim.x * gridDim.x){
                __isnanf(@in0(i)) ? @out(i) = 0 : @out(i) = @in0(i);
            }
        }
        const int total_count = in0_shape0;
        const int thread_per_block = 1024;
        const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
        kernel1<<<block_count, thread_per_block>>>(@ARGS);
    """

    return jt.code(shape=input.shape, dtype=input.dtype, inputs=[input], cuda_header=CUDA_HEADER, cuda_src=CUDA_SRC)

def accumulate(input):
    CUDA_HEADER = r'''
    #include <cmath>
    using namespace std;
    '''

    CUDA_SRC = r"""
        __global__ static void kernel1(@ARGS_DEF) {
            @PRECALC
            float max = 0;
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < in0_shape0; i += blockDim.x * gridDim.x){
                max > @in0(i) ? @out(i) = max : max = @in0(i);
            }
        }
        const int total_count = in0_shape0;
        const int thread_per_block = 1024;
        const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
        kernel1<<<block_count, thread_per_block>>>(@ARGS);
    """

    return jt.code(shape=input.shape, dtype=input.dtype, inputs=[input], cuda_header=CUDA_HEADER, cuda_src=CUDA_SRC)

def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': nanmean(ap)}

def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = jt.zeros(gt_bbox.shape[0], dtype='bool')

        for l in jt.unique(jt.concat([pred_label, gt_label]).astype('int8')):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += jt.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(dim=1)[0]
            gt_index[iou.max(dim=1) < iou_thresh] = -1
            del iou

            selec = jt.zeros(gt_bbox_l.shape[0], dtype='bool')
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = jt.array(score[l])
        match_l = jt.array(match[l], dtype='int8')

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = jt.cumsum(match_l == 1)
        fp = jt.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)
    ap = jt.empty((n_fg_class, ))
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = float('nan')
            continue

        if use_07_metric:
            ap[l] = 0
            for t in jt.arange(0., 1.1, 0.1):
                if jt.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = jt.max(nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            mpre = jt.concat([[0], nan_to_num(prec[l]), [0]], dim=0)
            mrec = jt.concat([[0], rec[l], [1]], dim=0)

            mpre = accumulate(jt.maximum(mpre[::-1])[::-1])


            i = jt.where(mrec[1:] != mrec[:-1])[0]

            ap[l] = jt.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap