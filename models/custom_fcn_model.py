import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils, keys


class CustomFCN(nn.Module):

    def __init__(self, opt):
        super(CustomFCN, self).__init__()
        self.opt = opt
        self._make_model()
        if self.opt.load_weights:
            self.load_pretrained_weights()

    def _make_model(self):
        if self.opt.learning_type == keys.GUIDED_COMPLETION:
            num_input_channels = 2 + 3
        elif self.opt.learning_type == keys.UNGUIDED_COMPLETION:
            num_input_channels = 2
        self.conv_1 = nn.Conv2d(num_input_channels, 32, 5, 2)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 64, 3, 1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.score_1 = nn.Conv2d(64, 64, 1, 1)
        self.conv_3 = nn.Conv2d(64, 64, 3, 2)
        self.bn_3 = nn.BatchNorm2d(64)
        self.score_2 = nn.Conv2d(64, 64, 1, 1)
        self.conv_4 = nn.Conv2d(64, 64, 3, 2)
        self.bn_4 = nn.BatchNorm2d(64)
        self.deconv_1 = nn.ConvTranspose2d(64, 64, 3, 2)
        self.bn_5 = nn.BatchNorm2d(64)
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 4, 2)
        self.bn_6 = nn.BatchNorm2d(64)
        self.deconv_3 = nn.ConvTranspose2d(64, 32, 6, 2)
        self.bn_7 = nn.BatchNorm2d(32)
        self.deconv_4 = nn.ConvTranspose2d(32, 1, 5, 1)
        self.soft_max = nn.Softmax2d()

    def forward(self, x):
        # encoding
        y = self.conv_1(x)
        y = self.bn_1(y)
        y = F.relu(y)
        y = self.conv_2(y)
        y = self.bn_2(y)
        y = F.relu(y)
        # feature map 1
        feat_map_1 = F.relu(self.score_1(y))

        y = self.conv_3(y)
        y = self.bn_3(y)
        y = F.relu(y)
        # feature map 2
        feat_map_2 = F.relu(self.score_2(y))

        y = self.conv_4(y)
        y = self.bn_4(y)
        y = F.relu(y)
        # decoding
        y = self.deconv_1(y)
        y = self.bn_5(y)
        y = F.relu(y)

        y = y + feat_map_2

        y = self.deconv_2(y)
        y = self.bn_6(y)
        y = F.relu(y)
        # y[:, :, 1:, 1:] = y[:, :, 1:, 1:] + feat_map_1
        y = y + feat_map_1

        y = self.deconv_3(y)
        y = self.bn_7(y)
        y = F.leaky_relu(y)
        y = self.deconv_4(y)
        # exec(utils.TEST_EMBEDDING)
        if self.opt.residual_learning:
            d_y = y[:, 0:1, :, :]
            d_x = x[:, 0:1, :, :]
            d = d_x + d_y
            # conf_x = x[:, 1:2, :, :]
            # conf_y = y[:, 1:2, :, :]
            # conf = torch.cat([conf_x, conf_y],dim=1)
            # conf = self.soft_max(conf)
            # d = torch.cat([d_x, d_y],dim=1)
            # d = d * conf
            # d = d.sum(keepdim=True, dim=1)
            # exec(utils.TEST_EMBEDDING)
        return d

    def load_pretrained_weights(self):
        # 1. load check point
        checkpoint_path = self._get_checkpoint_path()
        checkpoint = self._load_checkpoint(checkpoint_path)

        # 2. try loading weights
        key_name = 'model_state_dict'
        saved_weights = checkpoint[key_name]
        self.load_state_dict(saved_weights)

    def _get_checkpoint_path(self):
        if hasattr(self.opt, 'resume') and self.opt.resume:
            checkpoint_path = os.path.join(self.opt.out, 'checkpoint.pth.tar')
        else:
            if hasattr(self.opt, 'load_weights') and self.opt.load_weights:
                checkpoint_path = os.path.join(self.opt.out_dir, self.opt.load_weights, 'checkpoint.pth.tar')
        return  checkpoint_path

    def _load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
        else:
            print(checkpoint_path)
            raise FileNotFoundError('model check point cannnot found')
        return checkpoint
