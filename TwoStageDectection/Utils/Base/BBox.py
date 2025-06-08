import jittor as jt
from jittor import misc


def bbox_decode(src, loc):
    if src.shape[0] == 0:
        assert False, "!!!!"
        return jt.zeros((0, 4), dtype=loc.dtype)

    src_height = src[:, 2] - src[:, 0]
    src_width = src[:, 3] - src[:, 1]
    src_ctr_y = src[:, 0] + 0.5 * src_height
    src_ctr_x = src[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    # dh = jt.clamp(dh, -10.0, 10.0)
    # dw = jt.clamp(dw, -10.0, 10.0)
    # dy = jt.clamp(dy, -3.0, 10.0)
    # dx = jt.clamp(dx, -3.0, 10.0)

    ctr_y = dy * src_height[:].unsqueeze(1) + src_ctr_y[:].unsqueeze(1)
    ctr_x = dx * src_width[:].unsqueeze(1) + src_ctr_x[:].unsqueeze(1)
    h = jt.exp(dh) * src_height[:].unsqueeze(1)
    w = jt.exp(dw) * src_width[:].unsqueeze(1)


    dst_bbox = jt.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox_encode(src, dst):
    height = src[:, 2] - src[:, 0]
    width = src[:, 3] - src[:, 1]
    ctr_y = src[:, 0] + 0.5 * height
    ctr_x = src[:, 1] + 0.5 * width

    basey0 = dst[..., 0]
    basey1 = dst[..., 2]
    basex0 = dst[..., 1]
    basex1 = dst[..., 3]
    base_height = basey1 - basey0
    base_width = basex1 - basex0
    base_ctr_y = basey0 + 0.5 * base_height
    base_ctr_x = basex0 + 0.5 * base_width

    eps = misc.finfo(height.dtype).eps
    height = jt.maximum(height, eps)
    width = jt.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = jt.log(base_height / height)
    dw = jt.log(base_width / width)

    loc = jt.stack([dy, dx, dh, dw]).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    tl = jt.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = jt.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    wh = (br - tl).clamp(min_v=0)
    area_i = wh.prod(dim=2)

    area_a = (bbox_a[:, 2:] - bbox_a[:, :2]).prod(dim=1)
    area_b = (bbox_b[:, 2:] - bbox_b[:, :2]).prod(dim=1)

    union = area_a[:, None] + area_b - area_i

    return area_i / union


def generate_anchor_base(base_size=16, anchor_scales=None, ratios=None):

    if anchor_scales is None:
        anchor_scales = [8, 16, 32]

    if ratios is None:
        ratios = [0.5, 1, 2]

    py = base_size / 2.
    px = base_size / 2.

    anchor_base = jt.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype='float32')
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * jt.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * jt.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
