import torch
import torch.nn as nn
import torchvision

import time

class _RCNN_layer(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(_RCNN_layer, self).__init__()
        self.input_shape = input_shape

        self.fc = nn.Sequential(
            nn.Linear(self.input_shape[0] * self.input_shape[1] * self.input_shape[2], 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.cls = nn.Sequential(
            nn.Linear(4096, n_classes),
            nn.Softmax(dim=1)
        )
        self.bbox_reg = nn.Sequential(
            nn.Linear(4096, 4 * n_classes)
        )

    def forward(self, input):
        rcnn_start = time.time()
        flatten = input.view(-1, self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        fc_out = self.fc(flatten)
        cls_out = self.cls(fc_out)
        reg_out = self.bbox_reg(fc_out)
        rcnn_end = time.time()

        rcnn_time = rcnn_end - rcnn_start

        return cls_out, reg_out, rcnn_time



class _Feature_layer(nn.Module):
    def __init__(self, roipool_size=7):
        super(_Feature_layer, self).__init__()

        self.roipool_size = roipool_size

        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg16_conv = vgg16.features[:-1]
        self.reduction_ratio = self._reduction()

        self.roipool = nn.AdaptiveMaxPool2d((roipool_size,roipool_size))

    def forward(self, input, rois):
        """

        :param input: batch of image (batch_size, c, h, w)
        :param rois: batch of roi (batch_size, num_roi, 4)
        :return:
        """
        vgg16_start = time.time()
        features = self.vgg16_conv(input)
        device = features.device
        vgg16_end = time.time()

        roipool_start = time.time()
        # shape of post-roi pooling
        batch_size = rois.shape[0]
        num_roi = rois.shape[1]
        channel = features.shape[1]

        roi_features = torch.zeros(batch_size*num_roi,
                                   channel,
                                   self.roipool_size, self.roipool_size,
                                   dtype=features.dtype).to(device)

        # RoI
        for i,(feature,roi) in enumerate(zip(features,rois)):
            feature_width = feature.shape[2]
            feature_height = feature.shape[1]
            # print("feature shape : ", feature.shape)
            for j,r in enumerate(roi):
                # print(r)
                # print(torch.ceil(r * self.reduction_ratio).int())
                x, y, w, h = torch.ceil(r * self.reduction_ratio).int()
                w, h = max(w, self.roipool_size), max(h, self.roipool_size)
                x1, y1 = min(x, max(feature_width-w, 0)), min(y, max(feature_height-h, 0))
                x2, y2 = x1 + w, y1 + h

                # print(x1," ",x2," | ",y1," ",y2)
                #
                # print(feature[:,y1:y2,x1:x2].shape)

                roi_features[i*num_roi+j,:,:,:] = self.roipool(feature[:,y1:y2,x1:x2])

        roipool_end = time.time()

        vgg16_time = vgg16_end - vgg16_start
        roipool_time = roipool_end - roipool_start

        return roi_features, vgg16_time, roipool_time

    def _reduction(self):
        sample = torch.randn((1,3,256,256))
        feature = self.vgg16_conv(sample)

        reduction_ratio = feature.shape[2] / sample.shape[2]

        return reduction_ratio

class fastRCNN(nn.Module):
    def __init__(self, n_classes=81, roipool_size=7):
        super(fastRCNN, self).__init__()

        self.feature_layer = _Feature_layer(roipool_size=roipool_size)
        self.rcnn_layer = _RCNN_layer((512,roipool_size,roipool_size),n_classes)

    def forward(self, input, roi):
        roi_features, vgg16_time, roipool_time = self.feature_layer(input,roi)
        cls_out, reg_out, rcnn_time = self.rcnn_layer(roi_features)
        return cls_out, reg_out, vgg16_time, roipool_time, rcnn_time

