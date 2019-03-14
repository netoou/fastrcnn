import cv2
import numpy as np
import torch

from utils import xywhToxyxy
from utils import xyxyToxywh
from utils import iou

import copy



def selectiveSearchExtraction(img, search_type='fast'):
    """
    Selective Search wrapping function
    :param img: single input image (type of numpy array from cv2.imread)
    :param search_type: choice selective search type (fast, quality)
    :return: region proposals
    """
    img_width = img.shape[1]
    img_height = img.shape[0]

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    if search_type=='fast':
        ss.switchToSelectiveSearchFast()
    elif search_type=='quality':
        ss.switchToSelectiveSearchQuality()

    extracted_regions = ss.process()
    # change coordinates for cliping bounding boxes
    extracted_regions_xyxy = xywhToxyxy(extracted_regions)
    np.clip(extracted_regions_xyxy[:, 0], 0, img_width, out=extracted_regions_xyxy[:, 0])  # x1
    np.clip(extracted_regions_xyxy[:, 1], 0, img_height, out=extracted_regions_xyxy[:, 1])  # y1
    np.clip(extracted_regions_xyxy[:, 2], 0, img_width, out=extracted_regions_xyxy[:, 2])  # x2
    np.clip(extracted_regions_xyxy[:, 3], 0, img_height, out=extracted_regions_xyxy[:, 3])  # y2
    # change coordinates to original main coordinates
    extracted_regions = xyxyToxywh(extracted_regions_xyxy)

    return extracted_regions

def trainRoiExtract(extracted_regions,
                    batch_anns,
                    num_roi, pos_ratio=0.75):

    """
    Function for extract rois from regions
    give back selected rois and corresponding labels

    :param extracted_regions: (?, 4) tensor x, y, w, h
    :param batch_anns: (?, 5) tensor x, y, w, h, class
    :param num_roi: integer total roi number
    :param pos_ratio: float ratio of postive sample
    :return:
    """
    #print("extracted region shape : ", extracted_regions.shape)
    #print("batch_anns shape : ", batch_anns.shape)
    batch_size = batch_anns.shape[0]
    roi = torch.zeros(batch_size, num_roi, 4, dtype=torch.float)
    roi_label = torch.zeros(batch_size, num_roi, 5, dtype=torch.float)

    positive_size = int(round(num_roi * pos_ratio))
    #negative_size = num_roi - positive_size
    for batch_idx, (extracted_region, batch_ann) in enumerate(zip(extracted_regions, batch_anns)):
        ious = torch.tensor([[iou(box, bbox_gt) for bbox_gt in batch_ann] for box in extracted_region], dtype=torch.float)

        max_ious, max_ious_idx = torch.max(ious, dim=1)

        # torch.nonzero와 tensor 비교연산은 np.where와 비슷한 역할이 가능
        pos_idxs = torch.nonzero(max_ious >= 0.5).view(-1)
        if len(pos_idxs) < positive_size:
            #print("not enough positive rois")
            positive_size = len(pos_idxs)
        neg_idxs = torch.nonzero((max_ious > 0.1) * (max_ious < 0.5)).view(-1)

        if positive_size > 0:
            pos_idxs = np.random.choice(pos_idxs, size=positive_size, replace=False)
        #try:
        neg_idxs = np.random.choice(neg_idxs, size=num_roi - positive_size, replace=False)
        # except Exception as e:
        #     print(e)
        #     print(neg_idxs.shape)
        #     print(pos_idxs.shape)
        #     print(max_ious.shape)
        #     return -1, -1

        roi[batch_idx, :positive_size, :] = extracted_region[pos_idxs, :]
        roi[batch_idx, positive_size:, :] = extracted_region[neg_idxs, :]

        roi_label[batch_idx, :positive_size, :] = batch_ann[max_ious_idx[pos_idxs]]
        roi_label[batch_idx, positive_size:, :] = batch_ann[max_ious_idx[neg_idxs]]

    # roi_label : (x, y, w, h, cls)
    # print(roi_label)
    return roi, roi_label

def roiTarget(roi, roi_label, means_coef, stds_coef, numbered_category):
    # Normalization included
    # roi_label shape : (batch_size, num_roi, 5)
    # roi_label : (x, y, w, h, cls)
    gt_bbox, label = roi_label[:, :, :4], roi_label[:, :, 4]
    #print("before label : ",label)
    label.apply_(numbered_category.index)
    #print("after label : ",label)

    #print("means coef shape : ", means_coef.shape)
    #print("std coef shape : ", stds_coef.shape)
    mean_coef = torch.from_numpy(means_coef[label.long()]).float()
    std_coef = torch.from_numpy(stds_coef[label.long()]).float()

    gt_bbox = (gt_bbox - mean_coef) / std_coef
    roi = (roi - mean_coef) / std_coef

    gt_bbox = gt_bbox.repeat(1, 1, len(numbered_category))
    #roi = roi.repeat(1, 1, len(numbered_category))

    return roi, gt_bbox, label.long()


if __name__=='__main__':
    # testcode
    sample_region = torch.tensor([[47, 41, 12, 35],
                                  [81, 38, 36, 32],
                                  [40, 12, 30, 95],
                                  [15, 22, 80, 77],
                                  [20, 40, 20, 30],
                                  [30, 60, 50, 50],
                                  [10, 50, 60, 30]])
    sample_anns = torch.tensor([[10,10,20,20,1],
                                 [30,30,50,50,2],
                                 [50,20,30,10,3]])
    roi, roi_label = trainRoiExtract(sample_region,sample_anns, 4)







