'''this is the dataset with voronoi preprocessing
'''

import os
import re
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob

from datasets import baseline_dataset
from utils import utils, image_utils

class VoronoiKittiDepthDataset(baseline_dataset.BaseKittiDepthDataset):

    def __init__(self, opt, transform=None):
        super().__init__(opt, transform)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            return None
        sparse_img = Image.open(str(self.sparse_depth_data[idx]))
        gt_img = Image.open(str(self.gt_data[idx]))

        # convert to meters
        sparse_img = np.array(sparse_img, dtype=np.float32) / self.norm_factor
        gt_img = np.array(gt_img, dtype=np.float32) / self.norm_factor
        
        depth_vor, confi_map = image_utils.gen_voronoi_from_sparse(self.opt, sparse_img)

        # convert to torch tensor
        sparse_img = utils.to_torch(sparse_img[None])
        gt_img = utils.to_torch(gt_img[None])
        depth_vor = utils.to_torch(depth_vor[None])
        # normalize confidence map
        confi_map = utils.to_torch(confi_map[None]) / 255.0

        if self.load_rgb:
            rgb_img = Image.open(str(self.rgb_data[idx]))
            # normalize the color
            rgb_img = (np.array(rgb_img, dtype=np.float32) / 255.0) * 2.0 - 1.0
            rgb_img = utils.to_torch(np.transpose(rgb_img, (2, 0, 1)))

        
        if self.load_rgb:
            return sparse_img, depth_vor, confi_map, rgb_img, gt_img
        else:
            return sparse_img, depth_vor, confi_map, gt_img
