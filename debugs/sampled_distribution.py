# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
from arguments import config_parser
import os
os.environ["GLOG_minloglevel"] ="2"
import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader
from flare.dataset import DatasetLoader


def main(args, dataloader_train):

    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0
    
    progress_bar = tqdm.tqdm(range(epochs))

    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            print(views_subset['mp_blendshape'][..., 10], views_subset['mp_blendshape'][..., 11])


            if iter_ > 100:
                exit()

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=False)
    assert dataset_train.len_img == len(dataset_train.importance)
    dataset_sampler = torch.utils.data.WeightedRandomSampler(dataset_train.importance, dataset_train.len_img, replacement=True)
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate, drop_last=True, sampler=dataset_sampler)

    main(args, dataset_train)


