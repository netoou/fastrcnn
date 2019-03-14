import sys

liblist = ['/home/ailab/.pycharm_helpers/pydev',
 '/home/ailab/.pycharm_helpers/third_party/thriftpy',
 '/home/ailab/.pycharm_helpers/pydev',
 '/home/ailab/anaconda3/envs/PartialConv/lib/python37.zip',
 '/home/ailab/anaconda3/envs/PartialConv/lib/python3.7',
 '/home/ailab/anaconda3/envs/PartialConv/lib/python3.7/lib-dynload',
 '/home/ailab/anaconda3/envs/PartialConv/lib/python3.7/site-packages',
 '/home/ailab/.pycharm_helpers/pycharm_matplotlib_backend',
 '/home/ailab/anaconda3/envs/PartialConv/lib/python3.7/site-packages/IPython/extensions',
 '/tmp/pycharm_project_924',
 '/tmp/pycharm_project_924']

for libpath in liblist:
    if not libpath in sys.path:
        sys.path.append(libpath)

import torch
import numpy as np
import PIL

from roiExtraction import roiTarget

import os
import time


from dataset import COCOdb
from dataset import coco_collate
from fast_rcnn import fastRCNN
from roiExtraction import trainRoiExtract
from loss import MultiTaskLoss


os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATA_SIZE = 'full'  # datasize must be 'full' or integer
INPUT_SIZE = (512, 512)
NUM_EPOCH = 100
BATCH_SIZE = 1
NUM_ROI = 64
START_EPOCH = 0
PRINT_STEP = 100
CUDA = True
RESUME = False
NUM_WORKERS = 10
LR = 1e-4
TRAINING = True

DATADIR = '/data/coco'
DATATYPE = 'train2017'

SAVE_DIR = 'models/'
LOAD_PATH = ''
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if CUDA:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

if __name__ == '__main__':

    cocodb = COCOdb(dataDir=DATADIR, dataType=DATATYPE, input_size=INPUT_SIZE, train=TRAINING)
    coco_loader = torch.utils.data.DataLoader(cocodb, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)  #, collate_fn=coco_collate)
    print('dataloader generated')

    model = fastRCNN().to(DEVICE)
    criterion = MultiTaskLoss().to(DEVICE)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

    if RESUME:
        print("loading checkpoint %s" % (LOAD_PATH))
        checkpoint = torch.load(LOAD_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch']
        print("loaded checkpoint %s" % (LOAD_PATH))
        model.eval()

    num_iter = int(len(cocodb) / BATCH_SIZE)
    print("Start training!")

    for epoch in range(START_EPOCH, NUM_EPOCH):
        if TRAINING:
            model.train(mode=True)
        loss_temp = 0
        cls_loss_temp = 0
        reg_loss_temp = 0

        vgg16_time_temp = 0
        roipool_time_temp = 0
        rcnn_time_temp = 0

        epoch_start = time.time()
        step_start = time.time()

        coco_iter = iter(coco_loader)

        for step in range(num_iter):
            imgs, anns, regions = next(coco_iter)
            imgs, anns, regions = imgs.to(DEVICE), anns.to(DEVICE), regions.to(DEVICE)

            try:
                roi, roi_label = trainRoiExtract(regions, anns, num_roi=NUM_ROI)
            except Exception as e:
                print(e)
                continue

            # if roi == -1:
            #     # selective search cannot work well
            #     continue

            cls_out, reg_out, vgg16_time, roipool_time, rcnn_time = model(imgs, roi)

            # roi will be normalized at the next line
            roi, gt_bbox, label = roiTarget(roi, roi_label, cocodb.target_means, cocodb.target_stds,
                                            cocodb.numbered_category)
            roi, gt_bbox, label = roi.to(DEVICE), gt_bbox.to(DEVICE), label.to(DEVICE)
            # print("reg out shape", reg_out.shape)
            # print("roi shape : ", roi.shape)
            # print("roi repeat shape : ", roi.repeat(1, 1, len(cocodb.numbered_category)).shape)
            reg_out = reg_out + roi.repeat(1, 1, len(cocodb.numbered_category)).view(BATCH_SIZE*NUM_ROI, -1)

            gt_bbox = gt_bbox.view(BATCH_SIZE*NUM_ROI, -1)
            label = label.view(BATCH_SIZE*NUM_ROI)

            # print("roi shape : ", roi.shape)
            # print("gt bbox shape : ", gt_bbox.shape)
            # print("label shape : ", label.shape)

            loss, cls_loss, reg_loss = criterion(cls_out, label, reg_out, gt_bbox)

            loss_temp += loss.item()
            cls_loss_temp += cls_loss.item()
            reg_loss_temp += reg_loss.item()

            vgg16_time_temp += vgg16_time
            roipool_time_temp += roipool_time
            rcnn_time_temp += rcnn_time

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % PRINT_STEP == 0:
                step_end = time.time()
                if step > 0:
                    loss_temp /= (PRINT_STEP + 1)
                    cls_loss_temp /= (PRINT_STEP + 1)
                    reg_loss_temp /= (PRINT_STEP + 1)

                    vgg16_time_temp /= (PRINT_STEP + 1)
                    roipool_time_temp /= (PRINT_STEP + 1)
                    rcnn_time_temp /= (PRINT_STEP + 1)

                print("[epoch %2d][step %4d/%4d] loss : %.4f, cls_loss : %.4f,\
                 reg_loss : %.4f, time : %f, vgg16_time : %f,\
                  roipool_time : %f, rcnn_time : %f" \
                      % (epoch, step, num_iter, loss_temp, cls_loss_temp, reg_loss_temp, \
                         step_end - step_start, vgg16_time_temp, roipool_time_temp, rcnn_time_temp))

                loss_temp = 0
                cls_loss_temp = 0
                reg_loss_temp = 0

                vgg16_time_temp = 0
                roipool_time_temp = 0
                rcnn_time_temp = 0
                step_start = time.time()

            if step % int(num_iter/2) == 0:
                if step > 0:
                    save_path = os.path.join(SAVE_DIR, 'fast-rcnn_{}_{}.pth'.format(epoch, step))
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, save_path)
                    print('save model : {}'.format(save_path))

        epoch_end = time.time()
        # should validate model at every epoch
        print("epoch time : {}".format(epoch_end - epoch_start))

        if epoch % 1 == 0:
            save_path = os.path.join(SAVE_DIR, 'fast-rcnn_{}_{}.pth'.format(epoch, step))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
            print('save model : {}'.format(save_path))

    # print("roi shape : ", roi.shape)
    # print("gt bbox shape : ", gt_bbox.shape)
    # print("label shape : ", label.shape)
    # print("anns shape : ", anns.shape)
    # print("roi_label shape : ", roi_label.shape)
    # print("cls out shape : ", cls_out.shape)
    # print("reg out shape : ", reg_out.shape)
    # print("target means shape : ", cocodb.target_means.shape)
    # print("target stds shape : ", cocodb.target_stds.shape)




