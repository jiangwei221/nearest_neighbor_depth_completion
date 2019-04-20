import os
import math
import os.path as osp

import tqdm
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils import utils, metrics_utils, keys, image_utils
from trainers import base_trainer, tensorboard_helper


class GuidedFCNTrainer(base_trainer.BaseTrainer):
    '''the trainer for initial guess learning
    '''

    def __init__(self, opt, model, optimizer, criterion,
                 train_loader, val_loader):
        # init parent class
        super().__init__(opt, model, optimizer, criterion,
                         train_loader, val_loader)

    def validate_batch(self, data_pack):
        '''validate for one batch of data
        '''

        # print(img.shape)
        # exec(utils.TEST_EMBEDDING)
        assert self.model.training is False
        with torch.no_grad():
            filled_sparse_depth = data_pack[keys.NN_FILLED_SPARSE_DEPTH]
            sparse_confidence = data_pack[keys.NN_FILLED_SPARSE_CONFIDENCE]
            rgb_img = data_pack[keys.ALIGNED_RGB]
            target = data_pack[keys.ANNOTATED_DEPTH]
            # exec(utils.TEST_EMBEDDING)
            stacked_img = torch.cat([filled_sparse_depth, sparse_confidence, rgb_img], dim=1)
            out_map = self.model(stacked_img)
            valid_index = target > 0.1
            # exec(utils.TEST_EMBEDDING)
            loss = self.criterion(out_map[valid_index], target[valid_index])
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                # raise ValueError('loss is nan while validating')
                print('loss is nan while validating')
        return loss_data, out_map

    def validate(self):
        '''validate for whole validation dataset
        '''
        training = self.model.training
        self.model.eval()
        val_loss_list = []
        for batch_idx, data_pack in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            loss_data, out_map = self.validate_batch(data_pack)
            val_loss_list.append(loss_data)
        mean_loss = np.array(val_loss_list).mean()
        validation_data = {'val_loss': mean_loss,
                           'out_map': out_map, }
        self.push_validation_data(data_pack, validation_data)
        self.save_model()
        if training:
            self.model.train()

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
        }, osp.join(self.out, 'checkpoint.pth.tar'))

    def push_validation_data(self, data_pack, validation_data):
        val_loss = validation_data['val_loss']
        rgb_img = vutils.make_grid(data_pack[keys.ALIGNED_RGB], normalize=True, scale_each=True)
        filled_sparse_depth = vutils.make_grid(data_pack[keys.NN_FILLED_SPARSE_DEPTH], normalize=True, scale_each=True)
        filled_gt_conf = vutils.make_grid(data_pack[keys.NN_FILLED_ANNOTATED_CONFIDENCE], normalize=True, scale_each=True)
        filled_gt_depth = vutils.make_grid(data_pack[keys.NN_FILLED_ANNOTATED_DEPTH], normalize=True, scale_each=True)
        out_map = vutils.make_grid(validation_data['out_map'], normalize=True, scale_each=True)
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(False)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_image({'NN input rgb': rgb_img})
        tb_datapack.add_image({'NN input depth': filled_sparse_depth})
        tb_datapack.add_image({'NN target': filled_gt_conf})
        tb_datapack.add_image({'NN target depth': filled_gt_depth})
        tb_datapack.add_image({'output': out_map})
        tb_datapack.add_scalar({'validation loss': val_loss})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def save_to_disk(self, data_pack, validation_data):
        for key in data_pack.keys():
            img = data_pack[key][0]
            img = image_utils.torch_img_to_np_img(img)
            try:
                plt.imsave(key, img)
            except:
                plt.imsave(key, img[:,:,0])

        img = validation_data['out_map'][0]
        img = image_utils.torch_img_to_np_img(img)
        try:
            plt.imsave('out_map', img)
        except:
            plt.imsave('out_map', img[:,:,0])

    def train_batch(self, data_pack):
        '''train for one batch of data
        '''
        assert self.model.training
        self.optim.zero_grad()
        filled_sparse_depth = data_pack[keys.NN_FILLED_SPARSE_DEPTH]
        sparse_confidence = data_pack[keys.NN_FILLED_SPARSE_CONFIDENCE]
        rgb_img = data_pack[keys.ALIGNED_RGB]
        target = data_pack[keys.ANNOTATED_DEPTH]
        stacked_img = torch.cat([filled_sparse_depth, sparse_confidence, rgb_img], dim=1)
        score = self.model(stacked_img)
        valid_index = target > 0.1
        loss = self.criterion(score[valid_index], target[valid_index])
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            print('loss is nan while training')
            exec(utils.TEST_EMBEDDING)
            exit(1)

        loss.backward()
        self.optim.step()
        self.push_training_data(data_pack, loss)

    def push_training_data(self, data_pack, loss):
        tb_datapack = tensorboard_helper.TensorboardDatapack()
        tb_datapack.set_training(True)
        tb_datapack.set_iteration(self.iteration)
        tb_datapack.add_scalar({'training loss': loss})
        self.tb_pusher.push_to_tensorboard(tb_datapack)

    def resume(self, opt):
        '''resume training:
        resume from the recorded epoch, iteration, and saved weights.
        resume from the model with the same name.
        
        Arguments:
            opt {[type]} -- [description]
        '''
        if hasattr(opt, 'load_weights'):
            assert opt.load_weights is None
        # 1. load check point
        checkpoint_path = os.path.join(opt.out, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            raise FileNotFoundError('model check point cannnot found: {0}'.format(checkpoint_path))
        # 2. load data
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        self.model.load_pretrained_weights()
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
