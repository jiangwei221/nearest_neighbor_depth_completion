import time

import numpy as np
import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datasets import baseline_dataset, nn_filling_dataset

from utils import utils, metrics_utils, keys
from options import options

training = True

opt = options.set_options(training)
test_voronoi_dataset = nn_filling_dataset.NNFillingKittiDepthDataset(opt, keys.VALIDATION_DATA)
test_voronoi_dataloader = DataLoader(test_voronoi_dataset, shuffle=False, batch_size=opt.batch_size, num_workers=opt.workers)
mae = metrics_utils.MAE()
rmse = metrics_utils.RMSE()

mae_list = []
rmse_list = []
# exec(utils.TEST_EMBEDDING)
for batch_idx, data_pack in tqdm.tqdm(
                enumerate(test_voronoi_dataloader), 
                total=len(test_voronoi_dataloader), 
                ncols=80,
                leave=False):
    sparse_img, depth_vor, confi_map, rgb_img, gt_img = data_pack[keys.SPARSE_DEPTH], data_pack[keys.NN_FILLED_SPARSE_DEPTH], data_pack[keys.NN_FILLED_SPARSE_CONFIDENCE], data_pack[keys.ALIGNED_RGB], data_pack[keys.ANNOTATED_DEPTH]
    cur_mae_val = mae(depth_vor, gt_img)
    cur_rmse_val = rmse(depth_vor, gt_img)
    mae_list += [cur_mae_val.data.cpu().numpy()]
    rmse_list += [cur_rmse_val.data.cpu().numpy()]
    # if batch_idx == 129:
    #     break

exec(utils.TEST_EMBEDDING)
