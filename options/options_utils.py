'''utils for argparse
'''

import os
import sys
import json
from utils import utils


def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 'y', 't')


def print_opt(opt):
    content_list = []
    args = list(vars(opt))
    args.sort()
    for arg in args:
        content_list += [arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg))]
    utils.print_notification(content_list, 'OPTIONS')


def confirm_opt(opt):
    print_opt(opt)
    if not utils.confirm():
        exit(1)


def opt_to_string(opt) -> str:
    string = '\n\n'
    string += 'python ' + ' '.join(sys.argv)
    string += '\n\n'
    # string += '---------------------- CONFIG ----------------------\n'
    args = list(vars(opt))
    args.sort()
    for arg in args:
        string += arg.rjust(25, ' ') + '  ' + str(getattr(opt, arg)) + '\n\n'
    # string += '----------------------------------------------------\n'
    return string


def save_opt(opt):
    '''save options to a json file
    '''
    # exec(utils.TEST_EMBEDDING)
    with open(os.path.join(opt.out, 'params.json'), 'w') as fp:
        json.dump(vars(opt), fp, indent=0, sort_keys=True)
