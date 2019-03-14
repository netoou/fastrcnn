import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import gradcheck

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import os

from pycocotools.coco import COCO
from torchsummary import summary

print(torch.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="2"


class RPN(nn.Module):
    #####
    ##### 일단 batch_size를 1로 고정시켜놓고 실험
    #####
    def __init__(self, device, img_size=800, batch_size=1):
        super(RPN, self).__init__()
        # gt_bbox have to be processed to center x,y and w,h coordinates

        self.batch_size = batch_size
        self.training = False  # True# 일단 테스트용으로 False
        self.device = device

        self.img_size = img_size
        self.sub_sample = 16
        self.feature_size = (self.img_size // self.sub_sample)
        # anchor params
        self.ratios = [0.5, 1, 2]
        self.anchor_scales = [8, 16, 32]
        self.n_anchors = len(self.ratios) * len(self.anchor_scales)
        # sample params
        self.pos_ratio = 0.5
        self.n_sample = 256
        self.n_pos = self.pos_ratio * self.n_sample
        # NMS
        self.nms_thresh = 0.7
        self.n_train_pre_nms = 12000
        self.n_train_post_nms = 2000
        self.n_test_pre_nms = 6000
        self.n_test_post_nms = 300
        self.min_box_size = 16

        self.rpn_conv, self.rpn_reg, self.rpn_cls = self._network_init(512, self.n_anchors)

    def forward(self, in_features, gt_bboxes=None):
        rpn_net_time = time.time()
        total_n_anchors = self.n_anchors * self.feature_size * self.feature_size

        rpn_feature = self.rpn_conv(in_features)

        pred_rpn_reg = self.rpn_reg(rpn_feature)
        pred_rpn_reg = pred_rpn_reg.permute(0, 2, 3, 1).contiguous().view(self.batch_size, -1, 4)

        pred_rpn_cls = self.rpn_cls(rpn_feature)
        pred_rpn_cls = pred_rpn_cls.permute(0, 2, 3, 1).contiguous().view(self.batch_size, -1, 2)
        pred_rpn_cls = F.softmax(pred_rpn_cls, dim=2)
        rpn_net_time = round(time.time() - rpn_net_time, 2)
        # 0 : bg, 1 : fg

        rois = 0
        rpn_reg_loss = 0
        rpn_cls_loss = 0

        anchor_roi_time = time.time()

        anchors = self._generate_anchor(self.img_size,
                                        self.sub_sample,
                                        self.ratios,
                                        self.anchor_scales)

        rois = self._region_proposal(anchors, pred_rpn_reg, pred_rpn_cls)
        anchors = torch.from_numpy(anchors).to(self.device)
        anchor_roi_time = round(time.time() - anchor_roi_time, 2)

        rpn_loss_time = None
        #         print("rois shape : ", rois.shape)
        if self.training:
            rpn_loss_time = time.time()
            # valid_index : 이미지 경계선을 넘어가지 않는 anchor들의 index
            valid_index, rpn_reg_target, rpn_cls_target = self._rpn_targets(anchors, gt_bboxes)
            #             print("target function finished")
            # non crossing boarder
            # print(pred_rpn_reg.shape)
            valid_pred_reg = pred_rpn_reg[:, valid_index, :]
            valid_pred_cls = pred_rpn_cls[:, valid_index, :]
            #             print("valid_pred_reg shape : ", valid_pred_reg.shape)
            #             print("valid_pred_cls shape : ", valid_pred_cls.shape)
            # require_grad??????????
            rpn_reg_target = torch.from_numpy(rpn_reg_target).to(self.device)
            rpn_cls_target = torch.from_numpy(rpn_cls_target).to(self.device)
            #             print("rpn_reg_target shape : ", rpn_reg_target.shape)
            #             print("rpn_cls_target shape : ", rpn_cls_target.shape)

            anchors = anchors.expand(self.batch_size, -1, 4)

            # loss
            # reg 페이퍼에 나오는 식으로 바꿔서 로스 계산
            #             print("anchor device : ",anchors[:,valid_index,:].device)
            #             print("valid_pred_reg device : ",valid_pred_reg.device)
            #             print("rpn_reg_target device : ",rpn_reg_target.device)
            t_pred, t_gt = self._reg_loss_parameterization(anchors[:, valid_index, :],
                                                           valid_pred_reg,
                                                           rpn_reg_target)
            #             print("t_pred shape : ",t_pred.shape)
            #             print("t_gt shape : ",t_gt.shape)

            rpn_reg_loss = F.smooth_l1_loss(t_pred, t_gt, reduce=False)
            rpn_reg_loss = torch.mean(rpn_reg_loss, dim=2)
            rpn_reg_loss = rpn_reg_loss * rpn_cls_target.float()
            rpn_reg_loss = rpn_reg_loss.mean()
            # print("rpn_reg_loss shape : ",rpn_reg_loss.shape)

            rpn_cls_loss = []
            # cls
            for i_valid_pred_cls, i_rpn_cls_target in zip(valid_pred_cls, rpn_cls_target):
                batch_loss = F.cross_entropy(i_valid_pred_cls, i_rpn_cls_target, ignore_index=-1)
                rpn_cls_loss.append(batch_loss)

            rpn_cls_loss = torch.stack(rpn_cls_loss).mean()
            rpn_loss_time = round(time.time() - rpn_loss_time, 2)
            # print("rpn_cls_loss shape : ",rpn_cls_loss.shape)

        return rois, rpn_reg_loss, rpn_cls_loss, [rpn_net_time, anchor_roi_time, rpn_loss_time]

    def _reg_loss_parameterization(self, valid_anchors, valid_pred_reg, rpn_reg_target):
        # print("valid_anchors shape : ",valid_anchors.shape)
        # valid_pred_reg shape :  torch.Size([1, 14710, 4])
        # valid_pred_cls shape :  torch.Size([1, 14710, 2])
        # rpn_reg_target shape :  torch.Size([14710, 4])
        # rpn_cls_target shape :  torch.Size([14710])

        # before
        #         t_x_pred = (valid_pred_reg[:,:,0] - valid_anchors[:,:,0])/valid_anchors[:,:,2]
        #         t_y_pred = (valid_pred_reg[:,:,1] - valid_anchors[:,:,1])/valid_anchors[:,:,3]
        # after : valid 앵커 더해준거
        t_x_pred = (valid_pred_reg[:, :, 0]) / valid_anchors[:, :, 2]
        t_y_pred = (valid_pred_reg[:, :, 1]) / valid_anchors[:, :, 3]
        """
        w,h에서 nan값 발생
        아마도 zero division 때문에 nan값 발생
        w,h에서 나누기 연산이 있음
        아님

        log연산할 때 valid_pred_reg가 negative value가 나오면 로그함수 특성상 연산이 불가능함

        일단 pred에 원 anchor를 더하니 해결됨(pred는 앵커박스 전체를 예측하는게 아니라 원 앵커 박스에서 위치 세부조정)
        """
        # before
        t_w_pred = torch.log(valid_pred_reg[:, :, 2] / valid_anchors[:, :, 2])
        t_h_pred = torch.log(valid_pred_reg[:, :, 3] / valid_anchors[:, :, 3])
        # after : valid앵커 더해준거
        t_w_pred = torch.log((valid_pred_reg[:, :, 2] + valid_anchors[:, :, 2]) / valid_anchors[:, :, 2])
        t_h_pred = torch.log((valid_pred_reg[:, :, 3] + valid_anchors[:, :, 3]) / valid_anchors[:, :, 3])
        #         print("t_w_pred na? : ",True if int(torch.isnan(t_w_pred).max().data)==1 else False)
        #         print("t_h_pred na? : ",True if int(torch.isnan(t_h_pred).max().data)==1 else False)
        t_pred = torch.stack([t_x_pred, t_y_pred, t_w_pred, t_h_pred], dim=2)

        t_x_gt = (rpn_reg_target[:, :, 0] - valid_anchors[:, :, 0]) / valid_anchors[:, :, 2]
        t_y_gt = (rpn_reg_target[:, :, 1] - valid_anchors[:, :, 1]) / valid_anchors[:, :, 3]
        t_w_gt = torch.log(rpn_reg_target[:, :, 2] / valid_anchors[:, :, 2])
        t_h_gt = torch.log(rpn_reg_target[:, :, 3] / valid_anchors[:, :, 3])
        #         print("t_w_gt na? : ",True if int(torch.isnan(t_w_gt).max().data)==1 else False)
        #         print("t_h_gt na? : ",True if int(torch.isnan(t_h_gt).max().data)==1 else False)
        t_gt = torch.stack([t_x_gt, t_y_gt, t_w_gt, t_h_gt], dim=2)

        return t_pred, t_gt

    def _region_proposal(self, anchors, pred_rpn_reg, pred_rpn_cls):
        """
        지금 RoI가 이상하게 proposal되고 있음

        """
        batch_size = self.batch_size

        nms_thresh = self.nms_thresh
        n_pre_nms = self.n_train_pre_nms
        n_post_nms = self.n_train_post_nms
        if not self.training:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        min_box_size = self.min_box_size

        # ValueError: could not broadcast input array from shape (876,4) into shape (2000,4)

        # rois = np.zeros((batch_size, n_post_nms, 4))
        # coordinates : x1,y1 ,x2,y2
        for batch_idx, (pred_reg, pred_cls) in enumerate(zip(pred_rpn_reg, pred_rpn_cls)):
            np_pred_reg = pred_reg.cpu().data.numpy()
            np_pred_cls = pred_cls.cpu().data.numpy()
            # probability of foreground
            obj_prob = np_pred_cls[:, 1]
            #             print("np_pred_reg shape : ",np_pred_reg.shape)
            #             print("anchors shape : ",anchors.shape)
            roi = np.zeros(pred_reg.shape, dtype=np.float32)
            roi[:, 0] = np_pred_reg[:, 0] + anchors[:, 0]  # x1
            roi[:, 1] = np_pred_reg[:, 1] + anchors[:, 1]  # y1
            roi[:, 2] = np_pred_reg[:, 0] + anchors[:, 0] + np_pred_reg[:, 2] + anchors[:, 2]  # x2
            roi[:, 3] = np_pred_reg[:, 1] + anchors[:, 1] + np_pred_reg[:, 3] + anchors[:, 3]  # y2

            # clip predected boxes
            roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, self.img_size - 1)
            roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, self.img_size - 1)
            # remove small boxes (w or h < min_box_size)
            w = np_pred_reg[:, 2] + anchors[:, 2]
            h = np_pred_reg[:, 3] + anchors[:, 3]
            keeping_index = np.where((w >= min_box_size) & (h >= min_box_size))[0]
            roi = roi[keeping_index, :]
            obj_prob = obj_prob[keeping_index]
            # sorting to fg prob descending order
            descend_index = obj_prob.argsort()[::-1]
            # leave 12000 proposals before nms
            pre_nms_index = descend_index[:n_pre_nms]
            # roi = roi[pre_nms_index, :]
            # print(pre_nms_index.shape)
            # non-maximum suppression
            keep = []
            area = (roi[:, 2] - roi[:, 0] + 1) * (roi[:, 3] - roi[:, 1] + 1)
            while pre_nms_index.size > 0:
                idx = pre_nms_index[0]
                keep.append(idx)

                xx1 = np.maximum(roi[:, 0][idx], roi[:, 0][pre_nms_index[1:]])
                yy1 = np.maximum(roi[:, 1][idx], roi[:, 1][pre_nms_index[1:]])
                xx2 = np.minimum(roi[:, 2][idx], roi[:, 2][pre_nms_index[1:]])
                yy2 = np.maximum(roi[:, 3][idx], roi[:, 3][pre_nms_index[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)

                overlap = (w * h) / (area[idx] + area[pre_nms_index[1:]] - (w * h))
                non_suppression = np.where(overlap <= nms_thresh)[0]
                pre_nms_index = pre_nms_index[non_suppression + 1]

            keep = keep[:n_post_nms]
            # print(len(keep))
            # roi shape 0,4
            roi = roi[keep]
            # change coordinates
            roi[:, 2] = roi[:, 2] - roi[:, 0] + 1
            roi[:, 3] = roi[:, 3] - roi[:, 1] + 1
            rois = torch.tensor(torch.from_numpy(roi), dtype=torch.float32)

        return rois

    def _network_init(self, in_channels, n_anchors):

        rpn_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU())
        # reg output : 2(bg/fg) * (3*3 anchors)
        rpn_reg = nn.Conv2d(in_channels, 4 * n_anchors, kernel_size=1, stride=1, padding=0)
        # cls output : 4(x/y/w/h) * (3*3 anchors)
        rpn_cls = nn.Conv2d(in_channels, 2 * n_anchors, kernel_size=1, stride=1, padding=0)

        return rpn_conv, rpn_reg, rpn_cls

    @staticmethod
    def _generate_anchor(img_size,
                         sub_sample,
                         ratios,
                         anchor_scales):
        # 800x800이미지에서 50x50개의 총 2500개 anchor center point 생성
        feature_size = (img_size // sub_sample)

        ctr_x = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)
        ctr_y = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)
        ctr = np.zeros((feature_size * feature_size, 2))
        index = 0
        for x in ctr_x:
            for y in ctr_y:
                ctr[index, 1] = x - 8
                ctr[index, 0] = y - 8
                index += 1
        # anchor coordinates : top_left x, y, width, height

        anchors = np.zeros(((feature_size * feature_size * 9), 4), dtype=np.float32)
        index = 0
        for c in ctr:
            ctr_y, ctr_x = c
            for i, ratio in enumerate(ratios):
                for j, anchor_scale in enumerate(anchor_scales):
                    h = sub_sample * anchor_scale * np.sqrt(ratio)
                    w = sub_sample * anchor_scale * np.sqrt(1. / ratio)
                    anchors[index, 0] = ctr_x - w / 2.
                    anchors[index, 1] = ctr_y - h / 2.
                    anchors[index, 2] = w
                    anchors[index, 3] = h
                    index += 1

        return anchors

    def _rpn_targets(self, anchors, gt_bboxes, pos_threshold=0.7, neg_threshold=0.3):

        # valid anchors index
        anchors = anchors.cpu().data.numpy()
        valid_index = np.where((np.min(anchors, axis=1) >= 0) &
                               (np.max(anchors, axis=1) < 800))[0]

        ####
        ####
        # Assign objectiveness

        # set label for each anchor
        rpn_cls_target = np.empty((self.batch_size, len(valid_index)), dtype=np.long)
        rpn_cls_target.fill(-1)
        rpn_reg_target = np.zeros((self.batch_size, len(valid_index), 4), dtype=np.float32)

        # anchor filtered by anchor index
        valid_anchors = anchors[valid_index, :]
        # print("valid_anchors shape : ",valid_anchors.shape)

        # 8940개의 anchor와 bbox 2개에 대한 iou
        for k in range(self.batch_size):
            ious = np.zeros((len(valid_index), gt_bboxes.shape[1]), dtype=np.float32)
            #             print("ious shape : ", ious.shape)
            #             print("valid_anchors shape : ",valid_anchors.shape)
            for i, valid_anchor in enumerate(valid_anchors):
                for j, bbox in enumerate(gt_bboxes[k]):
                    #                     print("valid_anchor shape : ",valid_anchor.shape)
                    #                     print("bbox shape : ",bbox.shape)
                    #                     print("iou shape : ",iou(valid_anchor, bbox))
                    ious[i, j] = iou(valid_anchor, bbox)

            argmax_ious_axis0 = np.where(ious == ious.max(axis=0))[0]

            argmax_ious_axis1 = ious.argmax(axis=1)
            max_ious_axis1 = ious.max(axis=1)

            # assign objectiveness
            # classification target
            rpn_cls_target[k, max_ious_axis1 < neg_threshold] = 0
            rpn_cls_target[k, max_ious_axis1 >= pos_threshold] = 1
            rpn_cls_target[k, argmax_ious_axis0] = 1

            # bbox regression target
            rpn_reg_target[k, :, :] = gt_bboxes[k, argmax_ious_axis1]

        return valid_index, rpn_reg_target, rpn_cls_target

    def sampling_anchors_rpntrain(pos_ratio,
                                  n_sample,
                                  valid_anchors,
                                  objectiveness,
                                  objectiveness_bboxes):

        n_pos = pos_ratio * n_sample
        n_neg = n_sample - n_pos

        pos_index = np.where(objectiveness == 1)[0]
        if len(pos_index) > n_pos:
            pos_index = np.random.choice(pos_index, size=n_pos, replace=False)

        neg_index = np.where(objectiveness == 0)[0]
        if len(neg_index) > n_neg:
            neg_index = np.random.choice(neg_index, size=n_neg, replace=False)

    def anchor_loss_target(valid_anchors,
                           objectiveness_bboxes,
                           objectiveness,
                           epsilon=1e-6):

        # Ignore anchors that cross image boundaries

        # regression target
        valid_ctr_x = valid_anchors[:, 0]
        valid_ctr_y = valid_anchors[:, 1]
        valid_width = valid_anchors[:, 2]
        valid_height = valid_anchors[:, 3]

        obj_ctr_x = objectiveness_bboxes[:, 0]
        obj_ctr_y = objectiveness_bboxes[:, 1]
        obj_width = objectiveness_bboxes[:, 2]
        obj_height = objectiveness_bboxes[:, 3]

        valid_width = np.maximum(valid_width, epsilon)
        valid_height = np.maximum(valid_height, epsilon)

        tx = (obj_ctr_x - valid_ctr_x) / valid_width
        ty = (obj_ctr_y - valid_ctr_y) / valid_height
        tw = np.log(obj_width / valid_width)
        th = np.log(obj_height / valid_height)

        rpn_reg_target = np.vstack((tx, ty, tw, th)).transpose()

        ##################################

        # classification target
        rpn_cls_target = objectiveness

        return rpn_reg_target, rpn_cls_target
