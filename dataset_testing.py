import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import baseline_dataset, nn_filling_dataset
from utils import utils, metrics_utils, keys
from options import options

training = True

opt = options.set_options(training)

check_dataset = 'voronoi'

if check_dataset == 'base':
    test_base_dataset = baseline_dataset.BaseKittiDepthDataset(opt, keys.TRAINING_DATA)

    for idx in range(100):
        result_dict = test_base_dataset[idx]
        sparse_img, rgb_img, gt_img = result_dict[keys.SPARSE_DEPTH], result_dict[keys.ALIGNED_RGB], result_dict[keys.ANNOTATED_DEPTH]
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
    test_voronoi_dataset = nn_filling_dataset.NNFillingKittiDepthDataset(opt, keys.VALIDATION_DATA)

    for idx in range(100):
        print(idx)
        result_dict = test_voronoi_dataset[0+idx]
        sparse_img, depth_vor, confi_map, rgb_img = result_dict[keys.SPARSE_DEPTH], result_dict[keys.NN_FILLED_SPARSE_DEPTH], result_dict[keys.NN_FILLED_SPARSE_CONFIDENCE], result_dict[keys.ALIGNED_RGB]
        gt_img = result_dict[keys.ANNOTATED_DEPTH]
        gt_nn_depth = result_dict[keys.NN_FILLED_ANNOTATED_DEPTH]
        size = sparse_img.shape[::-1]

        visualize_imgs = True
        if visualize_imgs == True:
            # sparse_img
            try:
                f = plt.figure(figsize=(size[0]/100., size[1]/100.), dpi=100)
                plt.imshow(np.transpose(sparse_img.detach().cpu().numpy(), (1,2,0)))
            except:
                plt.imshow(np.transpose(sparse_img.detach().cpu().numpy(), (1,2,0))[:,:,0])
            plt.show()

            # rgb_img
            try:
                f = plt.figure(figsize=(size[0]/100., size[1]/100.), dpi=100)
                plt.imshow(np.transpose(rgb_img.detach().cpu().numpy()/2.0+0.5, (1,2,0)))
            except:
                plt.imshow(np.transpose(rgb_img.detach().cpu().numpy()/2.0+0.5, (1,2,0))[:,:,0])
            plt.show()

            # depth voronoi
            try:
                f = plt.figure(figsize=(size[0]/100., size[1]/100.), dpi=100)
                plt.imshow(np.transpose(depth_vor.detach().cpu().numpy(), (1,2,0)), cmap='gray')
            except:
                plt.imshow(np.transpose(depth_vor.detach().cpu().numpy(), (1,2,0))[:,:,0], )
            plt.show()

            # confidence map
            try:
                f = plt.figure(figsize=(size[0]/100., size[1]/100.), dpi=100)
                plt.imshow(np.transpose(confi_map.detach().cpu().numpy(), (1,2,0)), cmap='gray')
            except:
                plt.imshow(np.transpose(confi_map.detach().cpu().numpy(), (1,2,0))[:,:,0], cmap='gray')
            plt.show()

            # gt_img
            try:
                f = plt.figure(figsize=(size[0]/100., size[1]/100.), dpi=100)
                plt.imshow(np.transpose(gt_img.detach().cpu().numpy(), (1,2,0)))
            except:
                plt.imshow(np.transpose(gt_img.detach().cpu().numpy(), (1,2,0))[:,:,0])
            plt.show()

            # gt_nn_depth
            try:
                f = plt.figure(figsize=(size[0]/100., size[1]/100.), dpi=100)
                plt.imshow(np.transpose(gt_nn_depth.detach().cpu().numpy(), (1,2,0)))
            except:
                plt.imshow(np.transpose(gt_nn_depth.detach().cpu().numpy(), (1,2,0))[:,:,0])
            plt.show()

exec(utils.TEST_EMBEDDING)
