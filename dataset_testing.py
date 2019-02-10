import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import baseline_dataset, voronoi_dataset
from utils import utils
from options import options

training = False

opt = options.set_options(training)

check_dataset = 'voronoi'

if check_dataset == 'base':
    test_base_dataset = baseline_dataset.BaseKittiDepthDataset(opt)

    for idx in range(100):
        sparse_img, rgb_img, gt_img= test_base_dataset[idx]

        f = plt.figure()
        # sparse_img
        try:
            plt.imshow(np.transpose(sparse_img.detach().cpu().numpy(), (1,2,0)))
        except:
            plt.imshow(np.transpose(sparse_img.detach().cpu().numpy(), (1,2,0))[:,:,0])
        plt.show()

        # rgb_img
        try:
            plt.imshow(np.transpose(rgb_img.detach().cpu().numpy()/2.0+0.5, (1,2,0)))
        except:
            plt.imshow(np.transpose(rgb_img.detach().cpu().numpy()/2.0+0.5, (1,2,0))[:,:,0])
        plt.show()

        # gt_img
        try:
            plt.imshow(np.transpose(gt_img.detach().cpu().numpy(), (1,2,0)))
        except:
            plt.imshow(np.transpose(gt_img.detach().cpu().numpy(), (1,2,0))[:,:,0])
        plt.show()

elif check_dataset == 'voronoi':
    test_voronoi_dataset = voronoi_dataset.VoronoiKittiDepthDataset(opt)

    for idx in range(100):
        sparse_img, depth_vor, confi_map, rgb_img, gt_img = test_voronoi_dataset[12000+idx]

        f = plt.figure()
        # sparse_img
        try:
            plt.imshow(np.transpose(sparse_img.detach().cpu().numpy(), (1,2,0)))
        except:
            plt.imshow(np.transpose(sparse_img.detach().cpu().numpy(), (1,2,0))[:,:,0])
        plt.show()

        # rgb_img
        try:
            plt.imshow(np.transpose(rgb_img.detach().cpu().numpy()/2.0+0.5, (1,2,0)))
        except:
            plt.imshow(np.transpose(rgb_img.detach().cpu().numpy()/2.0+0.5, (1,2,0))[:,:,0])
        plt.show()

        # depth voronoi
        try:
            plt.imshow(np.transpose(depth_vor.detach().cpu().numpy(), (1,2,0)), cmap='gray')
        except:
            plt.imshow(np.transpose(depth_vor.detach().cpu().numpy(), (1,2,0))[:,:,0], cmap='gray')
        plt.show()

        # confidence map
        try:
            plt.imshow(np.transpose(confi_map.detach().cpu().numpy(), (1,2,0)), cmap='gray')
        except:
            plt.imshow(np.transpose(confi_map.detach().cpu().numpy(), (1,2,0))[:,:,0], cmap='gray')
        plt.show()

        # gt_img
        try:
            plt.imshow(np.transpose(gt_img.detach().cpu().numpy(), (1,2,0)))
        except:
            plt.imshow(np.transpose(gt_img.detach().cpu().numpy(), (1,2,0))[:,:,0])
        plt.show()

exec(utils.TEST_EMBEDDING)
