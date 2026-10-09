# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

import timm
import torchvision.transforms as transforms
import copy

from backbone import MammothBackbone, register_backbone

from backbone.vision_transformer_timm import vit_base_patch16_224_in21k_spt_EEE, vit_base_patch16_224_dino_EEE, vit_base_patch16_224_ibot_EEE

from timm.models import load_checkpoint

import torch.nn.functional as F

import torch.distributions as dist
from typing import Tuple, Literal

# num_classes_per_task = 40
#
# num_classes_all = 200

num_classes_per_task = 20

num_classes_all = 200

# num_classes_per_task = 40
#
# num_classes_all = 200

# num_classes_per_task = 10
#
# num_classes_all = 200

# num_classes_per_task = 10
#
# num_classes_all = 100

# num_classes_per_task = 345
#
# num_classes_all = 345 * 6


class eee(MammothBackbone):
    """
    eee network architecture.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Instantiates the layers of the network.
        """

        super(eee, self).__init__()

        self.device = 'cuda:2'

        #self.num_classes = num_classes

        self.model_dim = 768

        self.distributions = []

        self.c_expert = 0

        self.vitProcess = transforms.Compose(
        [transforms.Resize(224)])

        # model_name = 'vit_base_patch16_224_in21k_spt'
        # tuned_matrices = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        #                   [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0],
        #                   [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]]

        # tuned_matrices = [[0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0],
        #                   [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0],
        #                   [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]]

        tuned_matrices = [[1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                          [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]]

        # tuned_matrices = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        #                   [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        #                   [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]


        #sup
        self.vitmodel = vit_base_patch16_224_in21k_spt_EEE(img_size=224,
                                                 drop_rate=0.0,
                                                 drop_path_rate=0.0,
                                                 freeze_backbone=True,
                                                 structured_list=tuned_matrices,
                                                 # tuned_vectors=tuned_vectors,
                                                 low_rank_dim=10,
                                                 block='BlockSPTParallel',
                                                 num_classes=num_classes_all,
                                                 num_classes_per_task=num_classes_per_task,
                                                 structured_type='lora',
                                                 structured_bias=True,
                                                 no_structured_drop_out=False,
                                                 no_structured_drop_path=False,
                                                 )
        load_checkpoint(self.vitmodel, '/hy-tmp/model/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')

        # load_checkpoint(self.vitmodel, '/path/to/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')

        # #dino
        # self.vitmodel = vit_base_patch16_224_dino_EEE(img_size=224,
        #                                          drop_rate=0.0,
        #                                          drop_path_rate=0.0,
        #                                          freeze_backbone=True,
        #                                          structured_list=tuned_matrices,
        #                                          # tuned_vectors=tuned_vectors,
        #                                          low_rank_dim=10,
        #                                          block='BlockSPTParallel',
        #                                          num_classes=num_classes_all,
        #                                          num_classes_per_task=num_classes_per_task,
        #                                          structured_type='lora',
        #                                          structured_bias=True,
        #                                          no_structured_drop_out=False,
        #                                          no_structured_drop_path=False,
        #                                          )

        # # ibot
        # self.vitmodel = vit_base_patch16_224_ibot_EEE(img_size=224,
        #                                          drop_rate=0.0,
        #                                          drop_path_rate=0.0,
        #                                          freeze_backbone=True,
        #                                          structured_list=tuned_matrices,
        #                                          # tuned_vectors=tuned_vectors,
        #                                          low_rank_dim=10,
        #                                          block='BlockSPTParallel',
        #                                          num_classes=num_classes_all,
        #                                          num_classes_per_task=num_classes_per_task,
        #                                          structured_type='lora',
        #                                          structured_bias=True,
        #                                          no_structured_drop_out=False,
        #                                          no_structured_drop_path=False,
        #                                          )



        self.vitmodel.to(self.device)

        self.c_expert = 0

    def create_new_expert(self, tuned_matrices, past, seen, choose_idx=None):
        self.vitmodel.create_new_expert(tuned_matrices, past, seen, choose_idx=choose_idx)

        self.c_expert += 1

        self.vitmodel.to(self.device)

    def forward(self, x, preserve_sim=True, return_features=False):
        x = self.vitProcess(x)
        out = self.vitmodel(x, idx=self.c_expert, return_features=return_features)

        return out

    def myprediction(self, x, idx=None, return_features=False):
        with torch.no_grad():
            x = self.vitProcess(x)
            if idx is None:
                out = self.vitmodel(x, return_features=return_features)
            else:
                out = self.vitmodel(x, idx=idx, return_features=return_features)
            return out




@register_backbone("eee")
def eee_backbone(num_classes):
    return eee(num_classes)





