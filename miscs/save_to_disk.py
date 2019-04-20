import sys
import os
sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.misc

import tqdm
from datasets import nn_filling_dataset
from options import options
from utils import utils, keys

training = True

opt = options.set_options(training)
test_voronoi_dataset = nn_filling_dataset.NNFillingKittiDepthDataset(opt, keys.TRAINING_DATA)
test_voronoi_dataloader = DataLoader(test_voronoi_dataset,
                                     shuffle=False,
                                     batch_size=opt.batch_size,
                                     num_workers=opt.workers, )

# for batch_idx, data_pack in tqdm.tqdm(
#         enumerate(test_voronoi_dataloader),
#         total=len(test_voronoi_dataloader),
#         ncols=80,
#         leave=False):
#     sparse_img = data_pack[keys.SPARSE_DEPTH]
#     sparse_nn_depth = data_pack[keys.NN_FILLED_SPARSE_DEPTH]
#     sparse_confidence_map = data_pack[keys.NN_FILLED_SPARSE_CONFIDENCE]
#     gt_img = data_pack[keys.ANNOTATED_DEPTH]
#     gt_nn_depth = data_pack[keys.NN_FILLED_ANNOTATED_DEPTH]
#     gt_confidence_map = data_pack[keys.NN_FILLED_ANNOTATED_CONFIDENCE]
#     rgb_img = data_pack[keys.ALIGNED_RGB]
#     exec(utils.TEST_EMBEDDING)

for i in tqdm.tqdm(range(len(test_voronoi_dataset))):
    data_pack = test_voronoi_dataset[i]
    sparse_img = data_pack[keys.SPARSE_DEPTH]
    sparse_nn_depth = data_pack[keys.NN_FILLED_SPARSE_DEPTH]
    sparse_confidence_map = data_pack[keys.NN_FILLED_SPARSE_CONFIDENCE]
    gt_img = data_pack[keys.ANNOTATED_DEPTH]
    gt_nn_depth = data_pack[keys.NN_FILLED_ANNOTATED_DEPTH]
    gt_confidence_map = data_pack[keys.NN_FILLED_ANNOTATED_CONFIDENCE]
    rgb_img = data_pack[keys.ALIGNED_RGB]
    sparse_depth_path = data_pack['sparse_depth_path']
    gt_path = data_pack['gt_path']

    sparse_confidence_map = np.expand_dims(sparse_confidence_map, axis=2)
    gt_confidence_map = np.expand_dims(gt_confidence_map, axis=2)

    sparse_stitch = np.concatenate([sparse_nn_depth, sparse_confidence_map], axis=2)
    gt_stitch = np.concatenate([gt_nn_depth, gt_confidence_map], axis=2)

    sparse_depth_path = sparse_depth_path.replace('data_depth_velodyne', 'data_depth_velodyne_nn')
    gt_path = gt_path.replace('data_depth_annotated', 'data_depth_annotated_nn')
    os.makedirs(os.path.dirname(sparse_depth_path), exist_ok=True)
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    scipy.misc.imsave(sparse_depth_path, sparse_stitch)
    scipy.misc.imsave(gt_path, gt_stitch)

    # os.mkdir()
    # exec(utils.TEST_EMBEDDING)