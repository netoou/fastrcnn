import torch
import torchvision
import numpy as np

import cv2

import os

from pycocotools.coco import COCO
from PIL import Image

from utils import bbox_resize
from roiExtraction import selectiveSearchExtraction
from roiExtraction import trainRoiExtract

"""
Main coordinates of bbox is x,y,w,h
"""

class COCOdb(torch.utils.data.Dataset):
    def __init__(self, dataDir='/data/coco', dataType='train2017', dataSize='full', input_size=(512, 512), cat_type='full', train=True, ss_roi=True):
        super(COCOdb, self).__init__()
        # ss_roi : do selective search region proposal for fast rcnn

        assert len(input_size) == 2, "Dimension of input size should be 2."
        assert dataSize=='full' or type(dataSize)==int

        self.input_size = input_size
        self.ss_roi = ss_roi

        self.dataPath = os.path.join(dataDir, dataType)

        self.cocoTool, self.imgIds = self._init_cocotools(dataDir, dataType, dataSize, train)

        self.category = self.cocoTool.cats
        self.numbered_category = [0] + list(self.category.keys())

        self._rescale_bboxes(input_size)

        self.target_means, self.target_stds = self._target_normalize_coef()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=input_size),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        """
        :param index:
        :return: img, anns, extracted_region
        img : normalized image tensor
        anns : corresponding annotations (bbox, class)
        extarcted_regions : region porposals by selective search, if self.ss_roi == False then return None
        """
        imgId = self.imgIds[index]
        annIds = self.cocoTool.getAnnIds(imgId)

        filename = self.cocoTool.imgs[imgId]['file_name']
        path = os.path.join(self.dataPath, filename)
        # Image
        img = Image.open(path)
        img = self.transform(img.convert('RGB'))
        # Annotations
        anns = torch.zeros((len(annIds),5), dtype=torch.float)
        # [[x, y, w, h, category]
        #           ...
        #  [x, y, w, h, category]]
        for idx,annId in enumerate(annIds):
            ann = self.cocoTool.anns[annId]
            anns[idx,:4] = torch.tensor(ann['bbox'])
            anns[idx,4] = torch.tensor(ann['category_id'])
        # Extracted regions
        extracted_regions = None
        if self.ss_roi:
            cv_img = cv2.imread(path)
            cv_img = cv2.resize(cv_img, dsize=(self.input_size))
            extracted_regions = selectiveSearchExtraction(cv_img, search_type='fast')
            extracted_regions = torch.from_numpy(extracted_regions).float()

        return img, anns, extracted_regions

    def __len__(self):
        return len(self.imgIds)

    def _rescale_bboxes(self, input_size):
        # rescaling bounding boxes to corresponding input shape
        for imgId in self.imgIds:
            annIds = self.cocoTool.getAnnIds(imgId)

            img_width = self.cocoTool.imgs[imgId]['width']
            img_height = self.cocoTool.imgs[imgId]['height']

            img_size = (img_width, img_height)

            for annId in annIds:
                self.cocoTool.anns[annId]['bbox'] = \
                    bbox_resize(self.cocoTool.anns[annId]['bbox'], img_size, input_size)

    def _init_cocotools(self, dataDir, dataType, dataSize, train):
        dataDir = dataDir
        dataType = dataType
        dataSize = dataSize

        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[1]  # specify type here
        prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
        annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)

        cocoTool = COCO(annFile)

        imgIds = sorted(cocoTool.getImgIds())
        if not dataSize == 'full':
            imgIds = imgIds[0:dataSize]
        if not train:
            imgIds = imgIds[-dataSize:]

        return cocoTool, imgIds

    def _target_normalize_coef(self, eps=1e-8):

        cls_counts = np.zeros((len(self.category) + 1, 1)) + eps
        sums = np.zeros((len(self.category) + 1, 4))
        squared_sums = np.zeros((len(self.category) + 1, 4))

        for imgId in self.imgIds:
            annIds = self.cocoTool.getAnnIds(imgId)
            for annId in annIds:
                ann = self.cocoTool.anns[annId]
                category = ann['category_id']
                bbox = ann['bbox']
                cls_counts[self.numbered_category.index(category)] += 1
                sums[self.numbered_category.index(category), :] += bbox
                squared_sums[self.numbered_category.index(category), :] += bbox ** 2

        means = sums / cls_counts
        stds = np.sqrt(squared_sums / cls_counts - means ** 2)

        return means, stds

def coco_collate(batch):
    imgs = torch.stack([item[0] for item in batch], 0)
    anns = torch.stack([item[1] for item in batch], 0)
    regions = torch.stack([item[2] for item in batch], 0)
    return imgs, anns, regions

if __name__ == '__main__':
    cocodb = COCOdb()

    coco_loader = torch.utils.data.DataLoader(cocodb)

    a, a_anns, reg = cocodb[0]

    print("a_anns shape : ", a_anns.shape)
    print("a shape : ", a.shape)
    print("cocodb target means shape : ", cocodb.target_means.shape)
    print("cocodb target stds shape : ", cocodb.target_stds.shape)
    print("region proposals shape : ", reg.shape)

    roi, roi_label = trainRoiExtract(reg, a_anns, num_roi=10)

    print("roi")
    print(roi)
    print("roi_label")
    print(roi_label)
