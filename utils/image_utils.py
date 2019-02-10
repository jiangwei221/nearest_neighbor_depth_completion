import time
import numpy as np
import torch
from vispy import app
import matplotlib.pyplot as plt

from utils import utils
from renderer import voronoi_renderer

def gen_voronoi_from_sparse(opt, sparse_img):
    '''generate a voronoi diagram from a sparse depth image.
    each non-empty pixel will be considered as a vertex.
    
    Arguments:
        opt -- options.
        sparse_img -- sparse depth image, shape: (C, H, W), where C equals to 1.
                        if the shape is (H, W), add one dim to it.
    '''
    if len(sparse_img.shape) == 2:
        sparse_img = sparse_img[None]
    if len(sparse_img.shape) != 3 or sparse_img.shape[0] != 1:
        # exec(utils.TEST_EMBEDDING)
        raise ValueError('The input shape is wrong')
    thresh = 0.001
    C, H, W = sparse_img.shape
    # construct size, num_points, center, color
    size = (W, H)
    num_points =  np.count_nonzero(sparse_img)
    center = sparse_img.nonzero()[::-1][0:2]
    # extract depth value
    color = sparse_img[0, center[1], center[0]].astype(np.float16)
    # convert to uint8 array
    color = utils.split_float16_to_two_uint8(color)
    color = color.reshape((-1, 2))
    # append ones to B channel
    B_ones = np.expand_dims(np.ones_like(color[:,0]), axis=1) * 255
    color = np.concatenate([color, B_ones], axis=1).astype(np.float32) / 255.0
    center = np.transpose(np.array(center), (1, 0)).astype(np.float32)
    center[:,1] = H-center[:, 1]
    # render
    c = voronoi_renderer.Canvas(size=size, num_points=num_points, center=center, color=color, radius=opt.vor_radius)
    app.run()
    render = c.im

    # get data from rendered result
    depth_vor = render[:,:,0:2]
    confi_map = render[:,:,2]
    depth_vor= depth_vor.flatten()
    depth_vor = utils.merge_two_uint8_to_float16(depth_vor)
    depth_vor = depth_vor.reshape((H, W))
    depth_vor = depth_vor.astype(np.float32)
    confi_map = confi_map.astype(np.float32)

    return depth_vor, confi_map