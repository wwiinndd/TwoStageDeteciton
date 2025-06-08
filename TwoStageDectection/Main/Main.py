from __future__ import absolute_import
import os
import jittor as jt
import jittor.optim as optim
import jittor.dataset.dataset as jtdata
import jittor.lr_scheduler as lr_s

import sys
sys.path.append('/TwoStageDetection/')
from Utils.VOC_tools.VOCDataset import VOCDataset
from Networks.VGG16_FasterRCNN import FasterRCNNVGG16
from Utils.Base.Trainer import FasterRCNNTrainer
from Utils.Base.eval_tools import eval_detection_voc

from configparser import ConfigParser
from tqdm import tqdm
import resource
import os

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def resulteval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_)
        gt_labels += list(gt_labels_)
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        use_07_metric=True)
    return result


def main():
    jt.clean()

    # os.environ['JT_CHECK_NAN'] = '1'
    # os.environ['trace_py_var'] = '3'
    os.environ['debug'] = '3'
    os.environ['gdb_attach'] = '3'

    conf = ConfigParser()
    conf.read('PreConfig.ini', encoding='utf-8')

    if jt.has_cuda:
        jt.flags.use_cuda = 1

    datapath = conf['Path']['datapath']

    batch_size = conf.getint('train', 'batch_size')
    re_size = eval(conf['train']['crop_size'])
    num_workers = conf.getint('train', 'num_workers')
    epochs = conf.getint('train', 'epochs')
    rpn_sigma = conf.getfloat('train', 'rpn_sigma')
    roi_sigma = conf.getfloat('train', 'roi_sigma')
    test_num = conf.getint('train', 'test_num')

    lr = conf.getfloat('optimizer', 'lr')
    step_size = conf.getint('scheduler', 'step_size')
    gamma = conf.getfloat('scheduler', 'gamma')

    trainset = VOCDataset(data_dir=datapath, split='trainval', re_size=re_size)
    testset = VOCDataset(data_dir=datapath, split='test', re_size=re_size)

    print('load data')

    dataloader = jtdata.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = jtdata.DataLoader(testset, batch_size=1,  shuffle=False, num_workers=num_workers)

    model = FasterRCNNVGG16()
    print('model construct completed')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_s.StepLR(optimizer, step_size=step_size, gamma=gamma)

    trainer = FasterRCNNTrainer(model, rpn_sigma=rpn_sigma, roi_sigma=roi_sigma, optimizer=optimizer,
                                scheduler=scheduler)

    is_record = conf.getboolean("Record", "is_record")

    print("log")
    # if is_record:
    #     classname = conf.get("Record", "classname")
    #     sys.stdout = open('../Log/' + classname + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H_%M') + 'log.txt',
    #                       'w')

    for epoch in range(epochs):
        all_loss = 0
        all_num = 0
        for ii, (img, bbox_, label_) in tqdm(enumerate(dataloader), total=len(dataloader),
                                             desc=f"Epoch: {epoch+1}/{epochs} "):
            img, bbox, label = img.float(), bbox_, label_
            losses = trainer.train_step(img, bbox, label)
            all_loss += losses.total_loss
            all_num += img.shape[0]
        eval_result = resulteval(test_dataloader, model, test_num=test_num)
        log_info = 'epoch:{}, lr:{:.3e}, map:{:.3e},loss:{:.3e}'.format(epoch, (optimizer.param_groups[0]['lr']),
                                                  eval_result['map'],
                                                  trainer.get_meter_data())
        print(log_info)


if __name__ == '__main__':
    main()
    os.system("/usr/bin/shutdown")
