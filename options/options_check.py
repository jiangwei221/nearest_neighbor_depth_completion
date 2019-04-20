'''option checking methods
after get all the options, we check whether this
combination is valid or not
'''

import os
import re
from utils import utils

def check_existing(opt):
    existing = False
    if os.path.exists(opt.out) or os.path.exists(opt.tfb_out):
        content_list = []
        content_list += [opt.out, str(os.path.exists(opt.out))]
        content_list += [opt.tfb_out, str(os.path.exists(opt.tfb_out))]
        content_list += ['Found existing checkpoint and log']
        utils.print_notification(content_list, 'WARNING')
        existing = True
    else:
        content_list = []
        content_list += ['New model, no history']
        utils.print_notification(content_list)
        existing = False
    if existing is False and opt.resume is True:
        content_list = []
        content_list += ['No history, cannot resume']
        utils.print_notification(content_list, 'ERROR')
        exit(1)

def check_pretrained_weights(opt):
    if opt.load_weights:
        if hasattr(opt, 'resume') and opt.resume:
            content_list = []
            content_list += ['Resume or load weights, make your choice']
            utils.print_notification(content_list, 'ERROR')
            exit(1)
        existing = False
        weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
        if os.path.exists(weights_path):
            existing = True
        else:
            content_list = []
            content_list += ['Cannot find pretrained weights']
            utils.print_notification(content_list, 'ERROR')
            exit(1)