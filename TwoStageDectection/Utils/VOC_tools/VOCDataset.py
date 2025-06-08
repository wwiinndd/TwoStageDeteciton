import os
import xml.etree.ElementTree as ET
import jittor.nn as nn
import jittor as jt
from jittor.dataset import Dataset
from Utils.VOC_tools.data_utils import read_image


class VOCDataset(Dataset):

    def __init__(self, data_dir, split='trainval', re_size=None):
        super().__init__()


        if re_size is None:
            self.re_size = (600, 600)
        else:
            self.re_size = re_size

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.resize = nn.Resize(self.re_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = jt.stack(bbox).astype('float32')
        label = jt.stack(label).astype('int32')

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')

        img = read_image(img_file, color=True)
        img = jt.unsqueeze(img, dim=0)
        img = self.resize(img)
        img = jt.squeeze(img, dim=0)

        size = anno.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        scalex = self.re_size[0] / width
        scaley = self.re_size[1] / height
        for b in bbox:
            b[0] = int(b[0] * scaley)
            b[1] = int(b[1] * scalex)
            b[2] = int(b[2] * scaley)
            b[3] = int(b[3] * scalex)

        return img, bbox, label


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
