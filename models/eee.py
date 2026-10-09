# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch.nn.functional as F
import random
import copy


from itertools import chain, islice

import matplotlib.pyplot as plt
import math
import re
from collections import defaultdict

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


def make_positive_definite(covariance, epsilons=None):
    if epsilons is None:
        epsilons = [1e-9, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    n = covariance.size(0)
    identity = torch.eye(n, device=covariance.device)

    for eps in epsilons:
        cov_reg = covariance + eps * identity
        try:
            torch.linalg.cholesky(cov_reg)
            return cov_reg
        except:
            continue
    return cov_reg

def str2bool(v):
    print(v)
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ExpertAwareMomentTracker(nn.Module):

    def __init__(self, feat_dim: int,
                 n_task_per_expert: int = 2,
                 aggregation_strategy: str = 'weighted_avg',
                 ridge: float = 1e-3,
                 device: str = 'cuda',
                 reg_cur: bool = False,
                 reg_his: bool = False):
        super().__init__()

        self.feat_dim = feat_dim
        self.ridge = ridge
        self.device = device
        self.n_task_per_expert = n_task_per_expert
        self.aggregation_strategy = aggregation_strategy

        self.reg_cur = reg_cur
        self.reg_his = reg_his

        # ===== intra-expert =====
        self.current_expert_moments = []
        self.current_expert_task_count = 0

        # ===== inter-expert =====
        self.historical_expert_moments = []

        self.global_task_counter = 0

        self.training_task_active = True

    @torch.no_grad()
    def update_moment(self, features_batch: torch.Tensor) -> dict:
        """
        Args:
            features_batch: [B, D]

        Returns:
            metrics
        """
        if not self.training_task_active:
            B = features_batch.size(0)
            batch_M = torch.einsum('bd,be->de', features_batch, features_batch) / B
            batch_M = 0.5 * (batch_M + batch_M.T) + self.ridge * torch.eye(
                self.feat_dim, device=features_batch.device
            )

            metrics = {}
            self.current_expert_task_count += 1

            # ===== 1. update current moment=====
            if self.current_expert_task_count < self.n_task_per_expert:

                self.current_expert_moments.append(batch_M)
                metrics['task_moment_added'] = True
                self.global_task_counter += 1
                return metrics, self.current_expert_moments

            # ===== 2. aggregate current moment to a historical moment =====
            else:

                self.current_expert_moments.append(batch_M)
                metrics['task_moment_added'] = True


                M_archived = self._aggregate_current_expert_moments()
                self.historical_expert_moments.append(M_archived)

                # self.current_expert_moments = None
                temp = self.current_expert_moments
                self.current_expert_moments = []
                self.current_expert_task_count = 0
                metrics['expert_archived'] = True
                self.global_task_counter += 1

                return metrics, temp


    def force_aggregate_current_expert_moments(self) -> torch.Tensor:
        M_archived = self._aggregate_current_expert_moments()
        self.historical_expert_moments.append(M_archived)

        self.current_expert_moments = []
        self.current_expert_task_count = 0


    def _aggregate_current_expert_moments(self) -> torch.Tensor:
        """
        strategies：
        - 'mean'
        - 'weighted_avg'
        - 'max'
        """
        if not self.current_expert_moments:
            return None

        if self.aggregation_strategy == 'mean':
            M_agg = torch.stack(self.current_expert_moments).mean(dim=0)

        elif self.aggregation_strategy == 'weighted_avg':
            # weight by task id: 0, 1, 2 ...
            weights = torch.tensor(
                [i + 1 for i in range(len(self.current_expert_moments))],
                device=self.device, dtype=torch.float32
            )
            weights = weights / weights.sum()
            M_agg = sum(w * M for w, M in zip(weights, self.current_expert_moments))

        elif self.aggregation_strategy == 'max':
            # max
            M_agg = torch.stack(self.current_expert_moments).max(dim=0)[0]

        else:
            # default: mean
            M_agg = torch.stack(self.current_expert_moments).mean(dim=0)

        M_agg = 0.5 * (M_agg + M_agg.T) + self.ridge * torch.eye(
            self.feat_dim, device=self.device
        )

        return M_agg

    def _compute_single_loss(self, weights: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        cosine classifier moment loss
        """
        # 1. normalization
        weights_norm = F.normalize(weights, p=2, dim=1)  # key change!

        # 2. loss: trace(W_norm @ M @ W_norm^T)
        return ((weights_norm @ M) * weights_norm).sum()

    def forward(self, classifier_new_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            classifier_new_weights: [C_new, D]

        Returns:
            total_reg_loss
        """
        total_loss = 0.0
        count = 0

        # (W @ M) * W -> [C_new, D] -> sum
        if self.global_task_counter >= 1 and self.reg_his:
            # print(1)
            for his_M in self.historical_expert_moments:
                total_loss += self._compute_single_loss(classifier_new_weights, his_M)
                count += 1

        # traverse each moment belong to the current expert
        if self.reg_cur:
            for task_M in self.current_expert_moments:
                if task_M is not None:
                    total_loss += self._compute_single_loss(classifier_new_weights, task_M)
                    count += 1

        return total_loss / count if total_loss != 0.0 else 0.0

    def get_diagnostics(self) -> dict:
        """return diagnostics"""
        return {
            'current_expert_tasks': self.current_expert_task_count,
            'historical_experts': len(self.historical_expert_moments),
            'global_tasks': self.global_task_counter
        }


class eee(ContinualModel):
    NAME = 'eee'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:

        parser.add_argument('--reg_his', required=True,
                            help='if need regpast.')

        parser.add_argument('--reg_cur', required=True,
                            help='if need regcur.')

        parser.add_argument('--if_fallback', required=True,
                            help='if need fallback.')

        parser.add_argument('--if_gradproj', required=True,
                            help='if need proj.')

        parser.add_argument('--if_GAPP', required=True,
                            help='if need GAPP.')

        parser.add_argument('--if_partition', required=True,
                            help='if need partition.')

        parser.add_argument('--if_freeze_old_classes', required=True,
                            help='if need freeze old classes.')

        parser.add_argument('--task_per_exp', required=True,
                            help='number of task per expert.')

        parser.add_argument('--full_expansion', required=True,
                            help='if True, create a new expert for every task, ignoring capacity logic.')

        parser.add_argument('--importance_metric', type=str, default='magnitude',
                            help='Parameter importance metric: "magnitude" or "fisher".')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(eee, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.reg_his = str2bool(args.reg_his)
        self.reg_cur = str2bool(args.reg_cur)
        self.if_fallback = str2bool(args.if_fallback)
        self.if_gradproj = str2bool(args.if_gradproj)
        self.if_GAPP = str2bool(args.if_GAPP)
        self.if_partition = str2bool(args.if_partition)

        if self.if_partition and self.if_GAPP:
            self.group_strategy = 'lora+depth'
        else:
            self.group_strategy = 'no'

        self.if_freeze_old_classes = str2bool(args.if_freeze_old_classes)

        self.task_per_exp = int(args.task_per_exp)
        self.target_ratio = 1.0 / self.task_per_exp
        self.full_expansion = str2bool(args.full_expansion)
        self.importance_metric = args.importance_metric

        # dynamic expert capacity schedule state
        self.min_expert_capacity = 1  # min task count per expert
        # self.max_expert_capacity = getattr(args, 'max_task_per_exp', 4)  # max task count per expert
        self.current_expert_capacity = self.min_expert_capacity  # current expert schedule capacity
        self.expert_capacity_schedule = []  # record actual expert capacities [cap_0, cap_1, ...]

        self._next_expert_capacity_reduce = False  # if next expert needs capacity reduce
        # self.confusion_threshold = float(args.confusion_threshold)#1.0  # confusion threshold: R_curr/R_self > 1.0 is saturation



        self.train_loader_size = None
        self.iter = 0

        self.seen = 0
        self.past = 0

        self.cls_num = None

        self.cls_per_task = None

        self.store_masks = []

        self.gradient_mask_dict = None
        self.first_task = True

        self.entropy_list = []
        self.sum_entropy = None

        self.remaining_lora_ranks = 10




        self.all_task_features = {}

        self.plot_feature = False

        self.squence_name = args.dataset

    def _compute_grouped_threshold(self, scores_dict,
                                   group_strategy='lora+depth'):
        """
        Group-Aware Parameter Partitioning threshold

        Args:
            scores_dict: {param_name: tensor}
            group_strategy:
                - 'global'
                - 'lora'
                - 'depth'
                - 'lora+depth'
        """

        groups = defaultdict(list)

        for name, tensor in scores_dict.items():
            group_key = 'other'

            # 1. LoRA group
            if group_strategy in ['lora', 'lora+depth']:
                if 'learnable.1' in name:
                    group_key = 'lora_up'
                elif 'learnable.0' in name:
                    group_key = 'lora_down'

            # 2. ViT depth group
            if group_strategy in ['depth', 'lora+depth'] and 'blocks.' in name:
                match = re.search(r'blocks\.(\d+)\.', name)
                if match:
                    block_idx = int(match.group(1))
                    # shallow (0-5) / deep (6-11)
                    depth_group = 'shallow' if block_idx < 6 else 'deep'
                    # group key: 'lora_up_shallow', 'lora_down_deep' etc.
                    if group_key != 'other':
                        group_key = f'{group_key}_{depth_group}'
                    else:
                        group_key = f'vit_{depth_group}'

            groups[group_key].append(tensor)

        # ===== each group separately calculate threshold =====
        thresholds = {}
        for group_key, tensors in groups.items():
            if tensors:
                group_vals = torch.cat([t.flatten() for t in tensors])
                thresholds[group_key] = torch.quantile(group_vals, 1 - self.target_ratio)

        return thresholds  # dict: {group_key: threshold_value}

    def _generate_grouped_mask(self, scores_dict, thresholds):
        """
        generate gradient mask according to group thresholds
        """
        gradient_mask_dict = {}

        for name, tensor in scores_dict.items():

            group_key = self._get_param_group_key(name)


            threshold = thresholds.get(group_key, thresholds.get('global', None))
            if threshold is None:

                all_vals = torch.cat([v.flatten() for v in scores_dict.values()])
                threshold = torch.quantile(all_vals, self.target_ratio)


            mask = (tensor.abs().detach() >= threshold).float()
            gradient_mask_dict[name] = mask

        return gradient_mask_dict

    def _get_param_group_key(self, param_name):
        """auxiliary function：extract group key of parameters"""

        group = 'other'

        # LoRA
        if 'learnable.1' in param_name:
            group = 'lora_up'
        elif 'learnable.0' in param_name:
            group = 'lora_down'

        # ViT depth
        if 'blocks.' in param_name:
            match = re.search(r'blocks\.(\d+)\.', param_name)
            if match:
                block_idx = int(match.group(1))
                depth = 'shallow' if block_idx < 6 else 'deep'
                group = f'{group}_{depth}' if group != 'other' else f'vit_{depth}'

        return group

    def _evaluate_expert_capacity(self, moments_list, top_k=32, beta=1.0) -> bool:
        diagnostics = {}

        # ===== metric: response =====
        # calculate the cross response between each historical moment to other moments
        responses = []
        for i, M_i in enumerate(moments_list):
            task_responses = []
            for j, M_j in enumerate(moments_list):
                if i == j:
                    continue
                # M_i to M_j
                try:
                    eigvals_j, eigvecs_j = torch.linalg.eigh(M_j)

                    k = min(16, eigvals_j.size(0) // 4)
                    principal_j = eigvecs_j[:, -k:]  # [D, k]

                    # tr(P_j^T M_i P_j)
                    resp = torch.trace(principal_j.T @ M_i @ principal_j) / k
                    task_responses.append(resp.item())
                except:
                    task_responses.append(0.0)
            responses.append(task_responses)

        # calculate cross response ratio R_ij = response(M_i, task_j) / response(M_j, task_j)
        # Ideally, the moment of task i should exhibit the strongest response to task i itself.
        # Upon saturation, this "self-response advantage" diminishes.
        self_ratios = []
        cross_ratios = []

        for i in range(len(moments_list)):
            for j in range(len(moments_list)):
                if i == j:
                    continue
                # self response
                try:
                    eigvals_i, eigvecs_i = torch.linalg.eigh(moments_list[i])
                    k = min(16, eigvals_i.size(0) // 4)
                    principal_i = eigvecs_i[:, -k:]
                    self_resp = torch.trace(principal_i.T @ moments_list[i] @ principal_i) / k

                    # cross response
                    cross_resp = torch.trace(principal_i.T @ moments_list[j] @ principal_i) / k

                    ratio = cross_resp.item() / (self_resp.item() + 1e-8)
                    if i < j:
                        cross_ratios.append(ratio)
                    else:
                        self_ratios.append(ratio)
                except:
                    continue


        if len(cross_ratios) > 0 and len(self_ratios) > 0:
            avg_cross = np.mean(cross_ratios)
            avg_self = np.mean(self_ratios)
            # When cross_ratio approaches self_ratio, the tasks become difficult to distinguish.
            divergence = avg_cross / (avg_self + 1e-8)
            diagnostics['divergence'] = divergence
            diagnostics['avg_cross_ratio'] = avg_cross
            diagnostics['avg_self_ratio'] = avg_self
        else:
            divergence = 0.0

        print(f"[Divergence] Task {self.current_task + 1}: "
              f"avg_cross_ratio={diagnostics['avg_cross_ratio']:.3f}, "
              f"avg_self_ratio={beta * diagnostics['avg_self_ratio']:.3f}, "
              f"{'🔴 SATURATED' if diagnostics['avg_cross_ratio'] >= beta * diagnostics['avg_self_ratio'] else '🟢 OK'}")

        return diagnostics['avg_cross_ratio'] < beta * diagnostics['avg_self_ratio']



    def begin_epoch(self, epoch: int, dataset) -> None:

        if epoch != 0 and epoch != 5:
            return

        if epoch == 0 and not self.first_task:
            #self.gradient_mask_dict = {key: 1.0 - val for key, val in self.gradient_mask_dict.items()}

            keys = self.store_masks[0].keys()

            new_mask_dict = {}

            for key in keys:
                # [num_masks, dim1, dim2, ...]
                stacked_masks = torch.stack([mask_dict[key] for mask_dict in self.store_masks])


                union_mask = torch.max(stacked_masks, dim=0).values


                new_mask_dict[key] = 1.0 - union_mask


            self.gradient_mask_dict = new_mask_dict

        if epoch == 5:
            scores = {}

            attn_params = [(name, param) for name, param in self.net.vitmodel.named_parameters()
                           if "attn" in name and param.requires_grad]

            if self.importance_metric == 'fisher':
                train_loader = dataset.train_loader
                grad_sq_accum = {name: torch.zeros_like(param).to(self.device) for name, param in attn_params}
                total_samples = 0
                max_batches = 100
                count = 0

                self.net.vitmodel.zero_grad()
                for data in train_loader:
                    if count >= max_batches:
                        break
                    inputs = data[0].to(self.device)
                    labels = data[1].to(self.device)

                    outputs = self.net(inputs)
                    loss_ce = self.loss(outputs, labels)
                    loss_ce.backward()

                    for name, param in attn_params:
                        if param.grad is not None:

                            if not self.first_task and name in self.gradient_mask_dict:
                                mask = self.gradient_mask_dict[name].to(param.grad.device)
                                grad_sq_accum[name] += (param.grad.pow(2) * mask)
                            else:
                                grad_sq_accum[name] += param.grad.pow(2)

                    self.net.vitmodel.zero_grad()
                    total_samples += inputs.size(0)
                    count += 1

                epsilon = 1e-5
                for name in grad_sq_accum:
                    scores[name] = (grad_sq_accum[name] / total_samples) + epsilon
            else:
                for name, param in attn_params:
                    scores[name] = param.abs().detach()

            if not self.first_task:
                with torch.no_grad():
                    for name, val in scores.items():
                        if name in self.gradient_mask_dict:
                            mask = self.gradient_mask_dict[name]
                            scores[name] = val * mask

            # 1. calculate group thresholds
            thresholds = self._compute_grouped_threshold(
                scores,
                group_strategy=self.group_strategy  # 启用两级分组
            )

            # 2. generate group aware masks
            candidate_mask_dict = self._generate_grouped_mask(
                scores,
                thresholds
            )

            if not self.first_task:
                for key, val in candidate_mask_dict.items():
                    stacked_masks = torch.stack([candidate_mask_dict[key], self.gradient_mask_dict[key]])
                    inter_mask = torch.min(stacked_masks, dim=0).values
                    self.gradient_mask_dict[key] = inter_mask
            else:
                self.gradient_mask_dict = candidate_mask_dict

            self.store_masks.append(self.gradient_mask_dict)

            keys = self.store_masks[0].keys()

            new_mask_dict = {}

            for key in keys:

                #[num_masks, dim1, dim2, ...]
                stacked_masks = torch.stack([mask_dict[key] for mask_dict in self.store_masks])


                union_mask = torch.max(stacked_masks, dim=0).values

                new_mask_dict[key] = union_mask

            with torch.no_grad():
                for name, val in scores.items():
                    if name in new_mask_dict:
                        param_dict = dict(self.net.vitmodel.named_parameters())
                        if name in param_dict:
                            param = param_dict[name]
                            mask = new_mask_dict[name].to(param.device)
                            param.mul_(mask)




    def _normalize_scores_grouped(self, scores_dict: dict,
                                  eps: float = 1e-6) -> dict:
        """
        parameter dicts group-wise max normalization

        Args:
            scores_dict: {param_name: tensor}
            eps

        Returns:
            normalized: {param_name: tensor}
            group_max: {group_key: float}
        """
        # === Step 1: collect parameters for each group ===
        groups = defaultdict(list)
        for name, tensor in scores_dict.items():
            group_key = self._get_param_group_key(name)
            groups[group_key].append((name, tensor))

        # === Step 2: Max normalization for each group ===
        normalized = {}
        group_max = {}

        for group_key, items in groups.items():
            if not items:
                continue


            names, tensors = zip(*items)


            max_val = max(t.max().item() for t in tensors)
            max_val = max(max_val, eps)
            group_max[group_key] = max_val

            # score / max_group
            for name, tensor in zip(names, tensors):
                normalized[name] = tensor / max_val

        return normalized, group_max

    def end_task(self, dataset, eps=1e-8) -> None:

        if self.first_task:
            self.first_task = False


        # save distribution
        train_loader = dataset.train_loader
        num_choose = len(train_loader)

        max_batches = 100

        self.cur_tracker.training_task_active = False
        # self.net.eval()
        with torch.no_grad():
            train_iter = iter(train_loader)

            pbar = tqdm(train_iter, total=num_choose,
                        desc=f"Calculate distribution for task {self.current_task + 1}",
                        disable=False, mininterval=0.5)

            fc_features_list = []
            #task_moment = torch.zeros_like(self.cur_tracker.m_hist)
            count = 0
            #total_samples = 0
            while count < num_choose:
                try:
                    data = next(train_iter)
                except StopIteration:
                    break

                # if self.debug and count == 10:
                #     break

                x = data[0]
                x = x.to(self.device)

                processX = self.net.vitProcess(x)

                _, features = self.net.myprediction(processX, idx=self.net.c_expert,
                                                    return_features=True)  # self.net(processX, return_features=True)
                features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-8)
                fc_features_list.append(features)

                # if count == max_batches - 1 or count == num_choose - 1:
                #     if self.plot_feature:
                #         fc_features_tsne = torch.cat(fc_features_list, dim=0)
                #         self.all_task_features[self.current_task] = fc_features_tsne
                #         # if self.current_task == 9:
                #         self.visualize_features()

                count += 1
                pbar.update()

            pbar.close()
            fc_features = torch.cat(fc_features_list, dim=0)  # [num*b,fc_size]

            # update moments
            metrics, saved_cur_m_list = self.cur_tracker.update_moment(fc_features)

            # judge capacity
            expert_start_task = sum(self.expert_capacity_schedule[:-1]) if len(self.expert_capacity_schedule) > 1 else 0
            task_idx_in_expert = self.current_task - expert_start_task

            if task_idx_in_expert >= 1 and not self._next_expert_capacity_reduce and self.if_fallback and not self.full_expansion:

                can_add_more = self._evaluate_expert_capacity(saved_cur_m_list)

                if not can_add_more:
                    print(f"[Task {self.current_task + 1}] ⚠️ Expert capacity saturation detected! "
                          f"Next expert capacity will be reduced.")
                    self._next_expert_capacity_reduce = True
                    if task_idx_in_expert + 1 < self.expert_capacity_schedule[-1]:
                        self.cur_tracker.force_aggregate_current_expert_moments()
                        metrics['expert_archived'] = True
                        self.expert_capacity_schedule[-1] = task_idx_in_expert + 1
                        self.remaining_lora_ranks = 10

            if metrics:
                diag = self.cur_tracker.get_diagnostics()
                print(f"[Task {self.current_task + 1}] Tracker: {diag}, metrics: {metrics}")

            # self.net.train()
            self.cur_tracker.training_task_active = True


        # if (self.current_task + 1) % self.task_per_exp == 0:
        if self.current_task + 1 == sum(self.expert_capacity_schedule) or self._next_expert_capacity_reduce:
            self.net.vitmodel.freeze_stages(fine_tune_keywords=['head'])
        self.past += dataset.N_CLASSES_PER_TASK
        # pass



    def begin_task(self, dataset, threshold=0) -> None:
        if self.current_task == 0:
            init_capacity = 1 if self.full_expansion else self.min_expert_capacity

            self.cur_tracker = ExpertAwareMomentTracker(
                feat_dim=768,
                n_task_per_expert=self.min_expert_capacity,
                aggregation_strategy='mean',
                ridge=1e-3,
                device=self.device,
                reg_cur = self.reg_cur,
                reg_his = self.reg_his
            )
            # init capacity schedule
            self.current_expert_capacity = init_capacity#self.min_expert_capacity
            self.expert_capacity_schedule = [self.current_expert_capacity]
            print(f"[Task 1] Expert 1 capacity initialized: {self.current_expert_capacity}")



        train_loader = dataset.train_loader
        self.seen += dataset.N_CLASSES_PER_TASK
        self.offset_1, self.offset_2 = dataset.get_offsets(self.current_task)

        if self.cls_num is None:
            self.cls_num = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS #dataset.N_CLASSES
            self.cls_per_task = dataset.N_CLASSES_PER_TASK

        # dynamic create expert
        if self.current_task > 0:
            # judge if create new expert
            should_create_expert = False

            # if full expension
            if self.full_expansion:
                should_create_expert = True
            elif self._next_expert_capacity_reduce:
                should_create_expert = True
                print(f"[Task {self.current_task + 1}] ⚡ Force create new expert (capacity_reduce mode)")
            else:
                expert_start_task = sum(self.expert_capacity_schedule[:-1]) if len(
                    self.expert_capacity_schedule) > 1 else 0
                tasks_in_current_expert = self.current_task - expert_start_task
                if tasks_in_current_expert >= self.current_expert_capacity:
                    should_create_expert = True

            # create expert
            if should_create_expert:
                # ===== capacity adjust =====
                if self.full_expansion:
                    # full expansion capacity=1 forever
                    self.current_expert_capacity = 1
                    self._next_expert_capacity_reduce = False
                    print(f"[Task {self.current_task + 1}] ⚡ Full expansion mode: Capacity fixed to 1")
                elif self._next_expert_capacity_reduce:
                    self.current_expert_capacity = max(self.current_expert_capacity - 1, self.min_expert_capacity)
                    self._next_expert_capacity_reduce = False
                    print(f"[Task {self.current_task + 1}] ⚠️ Capacity reduced to {self.current_expert_capacity}")
                else:
                    self.current_expert_capacity += 1
                    print(f"[Task {self.current_task + 1}] Capacity promoted to {self.current_expert_capacity}")


                self.net.create_new_expert(tuned_matrices, self.past, self.seen)



                self.first_task = True
                self.store_masks = []

                # ===== record new expert attributes =====
                self.expert_capacity_schedule.append(self.current_expert_capacity)
                print(f"[Task {self.current_task + 1}] Expert {len(self.expert_capacity_schedule)} "
                      f"created with capacity: {self.current_expert_capacity}")

                # ===== synchro moment tracker =====
                self.cur_tracker.n_task_per_expert = self.current_expert_capacity
                self.target_ratio = 1.0 / self.current_expert_capacity

        self.opt = self.get_optimizer()

        opt_params = set()
        for group in self.opt.param_groups:
            for p in group['params']:
                opt_params.add(p)

        print(f"\n[Task {self.current_task + 1}] Optimizing parameters:")
        for name, param in self.net.named_parameters():
            if param in opt_params and param.requires_grad:
                print(f"  - {name} (shape: {param.shape})")

        total_trainable = sum(p.numel() for p in opt_params if p.requires_grad)
        print(f"Total trainable parameters: {total_trainable}\n")

    def myPrediction(self,x, idx=None, return_features=False):
        with torch.no_grad():

            out = self.net.myprediction(x, idx=idx, return_features=return_features)
            return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, fix_topd=16):
        l2_distance = torch.nn.MSELoss()

        self.opt.zero_grad()

        outputs, features = self.net(inputs, return_features=True)

        loss_sup = 0

        loss_ce = self.loss(outputs, labels)

        loss_vis = [loss_ce.item()]

        loss_tot = loss_ce

        if self.past > 0:
        # if self.past > 0 and self.sup_cur:
            new_weights = self.net.vitmodel.head.weight[self.past:self.seen, :]
            loss_sup += self.cur_tracker(new_weights)


        if loss_sup != 0:
            loss_vis.append(loss_sup.item())
            loss_tot += loss_sup

        loss_tot.backward()


        if self.past != 0 and self.if_freeze_old_classes:
            self.net.vitmodel.head.weight.grad.data[:self.past, :] = 0
            #self.net.vitmodel.head.bias.grad.data[:self.past] = 0


        if self.gradient_mask_dict != None and self.if_partition:
            for name, param in self.net.vitmodel.named_parameters():
                if name in self.gradient_mask_dict and self.gradient_mask_dict[name] is not None:
                    if param.grad is not None:
                        mask = self.gradient_mask_dict[name].to(param.grad.device)
                        param.grad.data = param.grad.data * mask

        if not self.first_task and self.if_gradproj:
            # ref_M = torch.stack(self.cur_tracker.current_expert_moments).mean(dim=0)
            #
            # D = ref_M.size(0)
            #
            # #P = I - V_top @ V_top.T
            # # ========================================================================
            # alpha_proj = 0.5  # projection strength
            #
            # # compute the full eigendecomposition to obtain the eigenvectors
            # eigvals_full, eigvecs_full = torch.linalg.eigh(ref_M)
            #
            # # Sort in descending order
            # eigvals_full = torch.flip(eigvals_full, dims=[0])
            # eigvecs_full = torch.flip(eigvecs_full, dims=[1])
            #
            # # extract top-k
            # top_k = fix_topd
            # V_top = eigvecs_full[:, :top_k]  #[D, top_k]
            #
            # # P = I - alpha * (V_top @ V_top.T)
            # identity = torch.eye(D, device=ref_M.device)
            # P = identity - alpha_proj * (V_top @ V_top.T)
            #
            # # Symmetrization
            # P = 0.5 * (P + P.T)

            ref_M = torch.stack(self.cur_tracker.current_expert_moments).mean(dim=0)

            # calculate projection matrix
            D = ref_M.size(0)
            eps = 1e-4
            alpha_proj = 0.5  # projection strength

            identity = torch.eye(D, device=self.device)
            #(M + εI)⁻¹
            inv_M = torch.linalg.solve(ref_M + eps * identity, identity)
            #P = I - α·M·(M+εI)⁻¹
            P = identity - alpha_proj * (ref_M @ inv_M)
            
            P = 0.5 * (P + P.T)

            # lora parameter gradient projection
            for name, param in self.net.vitmodel.named_parameters():
                if param.grad is None:
                    continue


                if 'learnable' in name.lower():
                    g = param.grad
                    if g.dim() == 2:
                        if g.shape[1] == D:
                            # lora_down [r, D] / head [C, D]
                            param.grad.data = g @ P
                        elif g.shape[0] == D:
                            # lora_up [D, r]
                            param.grad.data = P @ g


        self.opt.step()

        return loss_vis


