import numpy as np
from PIL import Image
import random
import jittor as jt
import jittor.misc as misc


def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        img = jt.array(img)
        return jt.unsqueeze(img, dim=0)
    else:
        # transpose (H, W, C) -> (C, H, W)
        img = jt.array(img)
        return jt.transpose(img, (2, 0, 1))