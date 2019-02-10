'''this is the datset with no preprocessing

image shape: 375*1242 = 465750

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

from utils import utils

class BaseKittiDepthDataset(Dataset):

    def __init__(self, opt, transform=None):
        self.opt = opt
        self.data_path = opt.data_path
        self.gt_path = opt.gt_path
        self.load_rgb = opt.load_rgb
        if self.load_rgb:
            self.rgb_path = opt.rgb_path
            self.rgb2gray = opt.rgb2gray
        self.setname = opt.setname
        self.transform = transform
        self.norm_factor = opt.norm_factor
        self.invert_depth = opt.invert_depth
        self.sparse_depth_data = list(sorted(glob.iglob(
                                            "{0}/{1}/**/*.png".format(self.data_path, self.setname), 
                                            recursive=True
                                            )))
        self.gt_data = list(sorted(glob.iglob(
                                            "{0}/{1}/**/*.png".format(self.gt_path, self.setname),
                                            recursive=True
                                            )))
        if self.load_rgb:
            self.rgb_data = []
            self._get_rgb_data()
        
        self._correspondence_check()

        if self.load_rgb:
            assert(len(self.gt_data) == len(self.sparse_depth_data) == len(self.rgb_data))
        else:
            assert(len(self.gt_data) == len(self.sparse_depth_data))


    def __len__(self):
        return len(self.sparse_depth_data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            return None
        sparse_img = Image.open(str(self.sparse_depth_data[idx]))
        gt_img = Image.open(str(self.gt_data[idx]))

        # convert to meters
        # TODO float16 or float32
        sparse_img = np.array(sparse_img, dtype=np.float32) / self.norm_factor
        gt_img = np.array(gt_img, dtype=np.float32) / self.norm_factor
        

        # convert to torch tensor
        sparse_img = utils.to_torch(sparse_img[None])
        gt_img = utils.to_torch(gt_img[None])

        if self.load_rgb:
            rgb_img = Image.open(str(self.rgb_data[idx]))
            # normalize the color
            rgb_img = (np.array(rgb_img, dtype=np.float32) / 255.0) * 2.0 - 1.0
            rgb_img = utils.to_torch(np.transpose(rgb_img, (2, 0, 1)))

        if self.load_rgb:
            return sparse_img, rgb_img, gt_img
        else:
            return sparse_img, gt_img

        # exec(utils.TEST_EMBEDDING)

    def _get_rgb_data(self):
        self.rgb_data = []
        for fname in self.sparse_depth_data:
            date_long = re.search('/{0}/(.*)/proj_depth/'.format(self.setname), fname)[1]
            date = date_long[:10]
            img_id = re.search('/velodyne_raw/(.*)', fname)[1]
            rgb_img_path = os.path.join(self.rgb_path, date, date_long, img_id.replace('/', '/data/'))
            if not os.path.isfile(rgb_img_path):
                content_list = []
                content_list += ['Cannot find corresponding RGB images']
                utils.print_notification(content_list, 'ERROR')
                exit(1)
            self.rgb_data.append(rgb_img_path)

    def _correspondence_check(self):
        if self.load_rgb:
            for sparse_img_path, gt_img_path, rgb_img_path in zip(self.sparse_depth_data,
                                                                    self.gt_data,
                                                                    self.rgb_data):
                # check img_id, camera_id, footage_id
                date_long = re.search('/{0}/(.*)/proj_depth/'.format(self.setname), sparse_img_path)[1]
                date = date_long[:10]
                img_id = re.search('/velodyne_raw/(.*)', sparse_img_path)[1]
                assert os.path.join(self.rgb_path, date, date_long, img_id.replace('/', '/data/')) == rgb_img_path
                assert sparse_img_path.replace('data_depth_velodyne', 'data_depth_annotated').replace('velodyne_raw', 'groundtruth') == gt_img_path
        else:
            for sparse_img_path, gt_img_path in zip(self.sparse_depth_data,
                                                    self.gt_data):
                # check img_id, camera_id, footage_id
                assert sparse_img_path.replace('data_depth_velodyne', 'data_depth_annotated').replace('velodyne_raw', 'groundtruth') == gt_img_path
        content_list = []
        content_list += ['Dataset is complete']
        utils.print_notification(content_list)