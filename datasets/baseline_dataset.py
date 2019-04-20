'''this is the datset with no preprocessing

original image shape: 375x1242 = 465750
crop image shape: 256x1216 = 311296

modified from nconv
https://github.com/abdo-eldesokey/nconv
original information:
########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################
'''

import os
import re
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob

from utils import utils, image_utils, keys


class BaseKittiDepthDataset(Dataset):

    def __init__(self, opt, dataset_type, transform=None):
        self.opt = opt
        self.data_path = opt.data_path
        self.gt_path = opt.gt_path
        self.load_rgb = opt.load_rgb
        if self.load_rgb:
            self.rgb_path = opt.rgb_path
            self.rgb2gray = opt.rgb2gray
        self.dataset_type = dataset_type
        self.transform = transform
        self.norm_factor = opt.norm_factor
        self.invert_depth = opt.invert_depth
        self.sparse_depth_paths = list(sorted(glob.iglob(
            "{0}/{1}/**/*.png".format(self.data_path, self.dataset_type),
            recursive=True
        )))
        self.gt_paths = list(sorted(glob.iglob(
            "{0}/{1}/**/*.png".format(self.gt_path, self.dataset_type),
            recursive=True
        )))
        # self._correspondence_check()
        if self.load_rgb:
            self._get_rgb_paths()

        if self.load_rgb:
            assert (len(self.gt_paths) == len(self.sparse_depth_paths) == len(self.rgb_paths))
        else:
            assert (len(self.gt_paths) == len(self.sparse_depth_paths))
        self.crop_height = 256
        self.crop_width = 1216
        self.base = 0

    def __len__(self):
        return len(self.sparse_depth_paths) // self.opt.data_cut

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            return None
        sparse_img = Image.open(str(self.sparse_depth_paths[idx]))
        gt_img = Image.open(str(self.gt_paths[idx]))

        # convert to meters
        # TODO float16 or float32
        sparse_img = np.array(sparse_img, dtype=np.float32) / self.norm_factor
        gt_img = np.array(gt_img, dtype=np.float32) / self.norm_factor
        sparse_img = self._crop_img(sparse_img)
        gt_img = self._crop_img(gt_img)

        # convert to torch tensor
        sparse_img = utils.to_torch(sparse_img[None])
        gt_img = utils.to_torch(gt_img[None])

        if self.load_rgb:
            rgb_img = self._read_rgb_image(str(self.rgb_paths[idx]))

        result_dict = {keys.SPARSE_DEPTH: sparse_img,
                       keys.ANNOTATED_DEPTH: gt_img, }

        if self.load_rgb:
            result_dict[keys.ALIGNED_RGB] = rgb_img
        return result_dict

    def _crop_img(self, input_img):
        # assert isinstance(input_img, np.ndarray)
        if len(input_img.shape) == 2:
            output_img = input_img[-self.crop_height:, self.base:self.base + self.crop_width]
        elif len(input_img.shape) == 3:
            output_img = input_img[-self.crop_height:, self.base:self.base + self.crop_width, :]
        else:
            raise ValueError()
        return output_img

    def _read_rgb_image(self, rgb_path: str):
        rgb_img = Image.open(rgb_path)
        # normalize the color
        rgb_img = (np.array(rgb_img, dtype=np.float32) / 255.0) * 2.0 - 1.0
        rgb_img = self._crop_img(rgb_img)
        rgb_img = utils.to_torch(np.transpose(rgb_img, (2, 0, 1)))
        rgb_img = image_utils.normalize_single_image(rgb_img)
        # exec(utils.TEST_EMBEDDING)
        return rgb_img

    def _get_rgb_paths(self):
        self.rgb_paths = []
        for fname in self.sparse_depth_paths:
            date_long = re.search('/{0}/(.*)/proj_depth/'.format(self.dataset_type), fname)[1]
            date = date_long[:10]
            img_id = re.search('/velodyne_raw/(.*)', fname)[1]
            rgb_img_path = os.path.join(self.rgb_path, date, date_long, img_id.replace('/', '/data/'))
            if not os.path.isfile(rgb_img_path):
                content_list = []
                content_list += ['Cannot find corresponding RGB images']
                utils.print_notification(content_list, 'ERROR')
                exit(1)
            self.rgb_paths.append(rgb_img_path)

    def _correspondence_check(self):
        if self.load_rgb:
            for sparse_img_path, gt_img_path, rgb_img_path in zip(self.sparse_depth_paths,
                                                                  self.gt_paths,
                                                                  self.rgb_paths):
                # check img_id, camera_id, footage_id
                date_long = re.search('/{0}/(.*)/proj_depth/'.format(self.dataset_type), sparse_img_path)[1]
                date = date_long[:10]
                img_id = re.search('/velodyne_raw/(.*)', sparse_img_path)[1]
                assert os.path.join(self.rgb_path, date, date_long, img_id.replace('/', '/data/')) == rgb_img_path
                assert sparse_img_path.replace('data_depth_velodyne', 'data_depth_annotated').replace('velodyne_raw',
                                                                                                      'groundtruth') == gt_img_path
                # ### once for all ###
                # sparse_img_shape = np.array(Image.open(sparse_img_path)).shape
                # gt_img_shape = np.array(Image.open(gt_img_path)).shape
                # rgb_img_shape = np.array(Image.open(rgb_img_path)).shape
                # if sparse_img_shape != (375, 1242) or gt_img_shape != (375, 1242) or rgb_img_shape != (375, 1242, 3):
                #     exec(utils.TEST_EMBEDDING)
                #     print(sparse_img_shape, gt_img_shape, rgb_img_shape)
                #     print(sparse_img_path, gt_img_path, rgb_img_path)
                # exec(utils.TEST_EMBEDDING)
        else:
            for sparse_img_path, gt_img_path in zip(self.sparse_depth_paths,
                                                    self.gt_paths):
                # check img_id, camera_id, footage_id
                assert sparse_img_path.replace('data_depth_velodyne', 'data_depth_annotated').replace('velodyne_raw',
                                                                                                      'groundtruth') == gt_img_path
        content_list = []
        content_list += ['Dataset is complete']
        utils.print_notification(content_list)
