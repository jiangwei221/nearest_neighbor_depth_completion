# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    import time
    import numpy as np
    import tqdm
    import torch
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt

    from models import custom_fcn_model
    from datasets import baseline_dataset, nn_filling_dataset
    from utils import utils, metrics_utils, keys
    from options import options
    from trainers import fcn_trainer

    training = True

    opt = options.set_options(training)
    train_nn_dataset = nn_filling_dataset.NNFillingKittiDepthDataset(opt, keys.TRAINING_DATA)
    train_nn_dataloader = DataLoader(train_nn_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.workers)

    val_nn_dataset = nn_filling_dataset.NNFillingKittiDepthDataset(opt, keys.VALIDATION_DATA)
    val_nn_dataloader = DataLoader(val_nn_dataset, shuffle=False, batch_size=opt.batch_size, num_workers=opt.workers)

    test_fcn = custom_fcn_model.CustomFCN(opt).cuda()
    criterion = torch.nn.MSELoss()
    optim_list = [{"params": test_fcn.parameters(), "lr": opt.learning_rate}]
    optim = torch.optim.Adam(optim_list)

    test_trainer = fcn_trainer.GuidedFCNTrainer(opt, test_fcn, optim, criterion, train_nn_dataloader, val_nn_dataloader)
    test_trainer.train()

    exec(utils.TEST_EMBEDDING)

if __name__ == '__main__':
    main()