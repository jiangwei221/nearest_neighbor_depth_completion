'''this is the dataset with voronoi preprocessing
'''

import os
import re
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob

from datasets import baseline_dataset
from utils import utils, image_utils, keys


class NNFillingKittiDepthDataset(baseline_dataset.BaseKittiDepthDataset):

    def __init__(self, opt, dataset_type, transform=None):
        super().__init__(opt, dataset_type, transform)
        self._load_nn_paths()
        self.thres = 0.5

    def _load_nn_paths(self):
        self.sparse_nn_depth_paths = list(sorted(glob.iglob(
            "{0}/{1}/**/*.png".format(self.data_path.replace('data_depth_velodyne', 'data_depth_velodyne_nn'),
                                      self.dataset_type),
            recursive=True
        )))
        self.gt_nn_paths = list(sorted(glob.iglob(
            "{0}/{1}/**/*.png".format(self.gt_path.replace('data_depth_annotated', 'data_depth_annotated_nn'),
                                      self.dataset_type),
            recursive=True
        )))

    def _read_nn_filled_image(self, nn_image_path: str):
        nn_img = Image.open(nn_image_path)
        nn_img = np.array(nn_img, dtype=np.uint8)
        depth = nn_img[:, :, 0:2]
        depth = self._RG_to_depth(depth)
        confidence_map = nn_img[:, :, 2]
        confidence_map = confidence_map.astype(np.float32)
        return depth, confidence_map

    def _RG_to_depth(self, depth):
        height, width, _ = depth.shape
        depth = depth.flatten()
        depth = utils.merge_two_uint8_to_float16(depth)
        depth = depth.reshape((height, width))
        depth = depth.astype(np.float32)
        return depth

    def __getitem__(self, idx):
        sparse_img = Image.open(str(self.sparse_depth_paths[idx]))
        gt_img = Image.open(str(self.gt_paths[idx]))
        sparse_nn_depth, sparse_confidence_map = self._read_nn_filled_image(str(self.sparse_nn_depth_paths[idx]))
        gt_nn_depth, gt_confidence_map = self._read_nn_filled_image(str(self.gt_nn_paths[idx]))
        # convert to meters
        sparse_img = np.array(sparse_img, dtype=np.float32) / self.norm_factor
        gt_img = np.array(gt_img, dtype=np.float32) / self.norm_factor

        sparse_img = self._crop_img(sparse_img)
        gt_img = self._crop_img(gt_img)

        # exec(utils.TEST_EMBEDDING)
        # sparse_nn_depth, sparse_confidence_map = image_utils.gen_voronoi_from_sparse(self.opt, sparse_img)
        # gt_nn_depth, gt_confidence_map = image_utils.gen_voronoi_from_sparse(self.opt, gt_img)
        # convert to torch tensor
        sparse_img = utils.to_torch(sparse_img[None])
        gt_img = utils.to_torch(gt_img[None])
        sparse_nn_depth = utils.to_torch(sparse_nn_depth[None])
        gt_nn_depth = utils.to_torch(gt_nn_depth[None])
        # normalize confidence map
        sparse_confidence_map = (utils.to_torch(sparse_confidence_map[None]) / 255.0) * 2.0 - 1.0
        gt_confidence_map = (utils.to_torch(gt_confidence_map[None]) / 255.0) * 2.0 - 1.0

        # exec(utils.TEST_EMBEDDING)
        result_dict = {keys.SPARSE_DEPTH: sparse_img,
                       keys.NN_FILLED_SPARSE_DEPTH: sparse_nn_depth,
                       keys.NN_FILLED_SPARSE_CONFIDENCE: sparse_confidence_map,
                       keys.ANNOTATED_DEPTH: gt_img,
                       keys.NN_FILLED_ANNOTATED_DEPTH: gt_nn_depth,
                       keys.NN_FILLED_ANNOTATED_CONFIDENCE: gt_confidence_map,
                       # 'sparse_depth_path': str(self.sparse_depth_paths[idx]),
                       # 'gt_path': str(self.gt_paths[idx])
                       }
        if self.load_rgb:
            rgb_img = self._read_rgb_image(str(self.rgb_paths[idx]))
            result_dict[keys.ALIGNED_RGB] = rgb_img

        if random.random() > self.thres:
            result_dict = self.flip_result_dict(result_dict)

        return result_dict

    def flip_result_dict(self, result_dict):
        for key in result_dict.keys():
            result_dict[key] = torch.flip(result_dict[key], [2])
        return result_dict
