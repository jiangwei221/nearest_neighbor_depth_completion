import time
import numpy as np
import torch
import cv2

DISABLE_EMBEDDING = False
TEST_EMBEDDING = '''
pass
''' if DISABLE_EMBEDDING else '''
import IPython
IPython.embed()
assert(0)
'''

def confirm(question='OK to continue?'):
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question + ' [y/n] ').lower()
    return answer == "y"

def print_notification(content_list, notifi_type='NOTIFICATION'):
    print(('---------------------- {0} ----------------------'.format(notifi_type)))
    print()
    for content in content_list:
        print(content)
    print()
    print('----------------------------------------------------')

def to_torch(nparray):
    tensor = torch.from_numpy(nparray).float().cuda()
    return torch.autograd.Variable(tensor, requires_grad=False)

def to_numpy(cudavar):
    return cudavar.data.cpu().numpy()

def isnan(x):
    return x != x

def hasnan(x):
    return isnan(x).any()

def split_float16_to_two_uint8(x):
    '''split float16 to two uint8
    
    Arguments:
        x {[type]} -- 1d np array
    '''
    assert x.dtype == np.float16, 'input type is not float16'
    assert len(x.shape) == 1, 'input is not an 1d array'
    return np.frombuffer(x.tobytes('C'), dtype=np.uint8)

def merge_two_uint8_to_float16(x):
    '''merge two uint8 to float16
    
    Arguments:
        x {[type]} -- [description]
    '''
    assert x.dtype == np.uint8, 'input type is not uint8'
    assert len(x.shape) == 1, 'input is not an 1d array'
    assert x.shape[0] % 2 == 0, 'the number of elemnets is not even'
    return np.frombuffer(x.tobytes('C'), dtype=np.float16)
