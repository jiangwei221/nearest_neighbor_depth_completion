import time
import numpy as np
import torch
from vispy import app
import matplotlib.pyplot as plt

from utils import utils
from renderer import voronoi_renderer


def squash_image(image):
    '''squash image to [0, 1]
    '''
    min_val = image.min()
    max_val = image.max()
    if isinstance(image, np.ndarray):
        image = np.interp(image, (min_val, max_val), (0, 1))
        return image
    elif isinstance(image, torch.Tensor):
        image_np = utils.to_numpy(image)
        image_np = np.interp(image_np, (min_val, max_val), (0, 1))
        image_torch = utils.to_torch(image_np)
        return image_torch
    else:
        raise ValueError('unsupported data type: {0}'.format(type(image)))


def normalize_single_image(image):
    assert len(image.shape) == 3
    img_mean = torch.mean(image, dim=(1, 2)).view(-1, 1, 1)
    img_std = image.contiguous().view(image.size(0), -1).std(-1).view(-1, 1, 1)
    image = (image - img_mean) / img_std
    return image


def torch_img_to_np_img(torch_img):
    '''convert a torch image to matplotlib-able numpy image
    torch use Channels x Height x Width
    numpy use Height x Width x Channels
    Arguments:
        torch_img {[type]} -- [description]
    '''
    if len(torch_img.shape) == 4 and (torch_img.shape[1] == 3 or torch_img.shape[1] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (0, 2, 3, 1))
    if len(torch_img.shape) == 3 and (torch_img.shape[0] == 3 or torch_img.shape[0] == 1):
        return np.transpose(torch_img.detach().cpu().numpy(), (1, 2, 0))
    elif len(torch_img.shape) == 2:
        return torch_img.detach().cpu().numpy()
    else:
        raise ValueError('cannot process this image')


def np_img_to_torch_img(np_img):
    '''convert a numpy image to bilinear-samplable torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width
    
    Arguments:
        np_img {[type]} -- [description]
    '''
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return utils.to_torch(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return utils.to_torch(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return utils.to_torch(np_img)
    else:
        raise ValueError('cannot process this image')


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
        raise ValueError('The input shape is wrong: {0}'.format(sparse_img.shape))
    channels, height, width = sparse_img.shape

    # construct size, num_points, center, color
    size = (width, height)
    num_points = np.count_nonzero(sparse_img)
    center = sparse_img.nonzero()[::-1][0:2]
    # extract depth value
    color = sparse_img[0, center[1], center[0]].astype(np.float16)
    # convert to uint8 array
    color = utils.split_float16_to_two_uint8(color)
    color = color.reshape((-1, 2))
    # append ones to B channel
    slice_of_ones = np.expand_dims(np.ones_like(color[:, 0]), axis=1) * 255
    color = np.concatenate([color, slice_of_ones], axis=1).astype(np.float32) / 255.0
    # color = np.random.uniform(0.0, 1.0, (num_points, 3))
    center = np.transpose(np.array(center), (1, 0)).astype(np.float32)
    # center[:,0] = W-center[:, 0]
    center[:, 1] = height - center[:, 1]
    # render
    c = voronoi_renderer.Canvas(size=size,
                                num_points=num_points,
                                center=center,
                                color=color,
                                radius=opt.vor_radius, )
    app.run()
    render = c.im

    # get data from rendered result
    depth_vor = render[:, :, 0:2]
    confidence_map = render[:, :, 2:]

    # depth_vor = depth_vor.flatten()
    # depth_vor = utils.merge_two_uint8_to_float16(depth_vor)
    # depth_vor = depth_vor.reshape((height, width))
    # depth_vor = depth_vor.astype(np.float32)

    # confidence_map = confidence_map.astype(np.float32)

    return depth_vor, confidence_map
