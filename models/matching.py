# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import torch.nn as nn
import numpy as np

from .superpoint import SuperPoint
from .superglue import SuperGlue
# from .sift import SIFT


class Matching(nn.Module):
    def __init__(self, config={}): # , use_sift=False):
        super().__init__()
        # self.use_sift = use_sift
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        # self.sift = SIFT() if use_sift else None
        self.superglue = SuperGlue(config.get('superglue', {}))
        # self.config = config

    def forward(self, data):
        pred = {}
        # if self.use_sift:
        #     if 'keypoints0' not in data:
        #         image0 = self.preprocess_image_for_sift(data['image0'])
        #         pred0 = self.sift({'image': image0})
        #         pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        #     if 'keypoints1' not in data:
        #         image1 = self.preprocess_image_for_sift(data['image1'])
        #         pred1 = self.sift({'image': image1})
        #         pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        # else:
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features for SuperGlue
        data = {**data, **pred}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching using SuperGlue
        pred = {**pred, **self.superglue(data)}

        return pred

    # def preprocess_image_for_sift(self, image_tensor):
    #     image = image_tensor.cpu().numpy().squeeze()
    #
    #     if image.max() <= 1.0:
    #         image = (image * 255).astype(np.uint8)
    #
    #     if image.ndim == 3 and image.shape[0] in [1, 3]:
    #         image = image.transpose(1, 2, 0)
    #         if image.shape[2] == 3:
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    #     return image

