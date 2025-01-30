#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Copyright (C) 2022  Gabriele Cazzato

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
'''

import random, re
from copy import deepcopy
from os import environ
import time
from datetime import timedelta
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

import datasets, models, optimizers, schedulers
from options import args_parser
from utils import average_updates, average_updates_mod, exp_details, get_acc_avg, printlog_stats, measure_model_size, average_s_map
from datasets_utils import Subset, TinyImageNetSubset, get_datasets_fig, get_datasets_fig_tiny_imagenet
from sampling import get_splits, get_splits_fig
from client import Client
import smap_utils

# modified, for HE
import os
import keygen
import he_utils as he_utils 
import openfhe
import tenseal as ts
from openfhe import *
# from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=10, test_interval=100)

def compare_weight_locations(top_weights_1, top_weights_2):
    """
    Compare two lists of weight locations to find common and different positions
    
    Args:
        top_weights_1, top_weights_2: Lists of tuples (name, weight_value, multi_idx, idx)
    
    Returns:
        dict with statistics and locations comparison
    """
    # Create sets of locations for each list using (name, multi_idx) as identifier
    locations_1 = {(name, multi_idx) for name, _, multi_idx, _ in top_weights_1}
    locations_2 = {(name, multi_idx) for name, _, multi_idx, _ in top_weights_2}
    
    # Find common and different locations
    common_locations = locations_1.intersection(locations_2)
    only_in_first = locations_1 - locations_2
    only_in_second = locations_2 - locations_1

    if len(locations_1) > 0:
        overlap_percentage = round(len(common_locations) / len(locations_1) * 100, 2)
    else:
        overlap_percentage = round(len(common_locations) / 1 * 100, 2) # if len(locations_1) ==0, meaning all parameters are used
    
    results = {
        "common_count": len(common_locations),
        "only_in_first_count": len(only_in_first),
        "only_in_second_count": len(only_in_second),
        "total_locations_1": len(locations_1),
        "total_locations_2": len(locations_2),
        "overlap_percentage": overlap_percentage,
        "common_locations": sorted(common_locations),
        "only_in_first": sorted(only_in_first),
        "only_in_second": sorted(only_in_second)
    }
    
    return results

import matplotlib.pyplot as plt
import matplotlib
def create_custom_colormap():
    """Create custom colormap for parameter importance visualization"""
    colors = ['black', 'red', 'green', 'yellow']
    return matplotlib.colors.ListedColormap(colors)


def visualize_importance_comparison(model, top_params_jac, top_params_mag, tboard_dir, suffix='comparison'):
    """
    Visualize parameter importance comparison. Shows black image if:
    1. No overlap exists between masks
    2. The layer name is not in model's state dict
    """
    # Get model's state dict keys
    state_dict_keys = set(model.state_dict().keys())
    
    # Create dictionaries to store binary masks for each method
    jac_masks = {}
    mag_masks = {}
    
    # Convert lists to masks
    for name, _, multi_idx, _ in top_params_jac:
        if name not in jac_masks and name in state_dict_keys:
            param_shape = None
            for n, p in average_global_map_jac.named_parameters():
                if n == name:
                    param_shape = p.shape
                    break
            jac_masks[name] = torch.zeros(param_shape)
        if name in jac_masks:  # Only add if the name was valid
            jac_masks[name][multi_idx] = 1
        
    for name, _, multi_idx, _ in top_params_mag:
        if name not in mag_masks and name in state_dict_keys:
            param_shape = None
            for n, p in average_global_map_mag.named_parameters():
                if n == name:
                    param_shape = p.shape
                    break
            mag_masks[name] = torch.zeros(param_shape)
        if name in mag_masks:  # Only add if the name was valid
            mag_masks[name][multi_idx] = 1
    
    # Process all unique layer names from both masks
    all_layer_names = set(jac_masks.keys()) | set(mag_masks.keys())
    
    for name in all_layer_names:
        # Check if the layer exists in state dict
        if name not in state_dict_keys:
            # Create a small black image for invalid layers
            plt.rcParams['font.size'] = 20
            fig, ax3 = plt.subplots(1, 1, figsize=(10, 10))
            black_image = np.zeros((10, 10))  # Small black image
            im3 = ax3.imshow(black_image, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax3.set_title(f'{name}\nLayer not in model')
            fig.savefig(os.path.join(tboard_dir, f'{name}_importance_{suffix}.png'))
            plt.close(fig)
            continue
            
        # Skip if layer is not in both masks
        if name not in jac_masks or name not in mag_masks:
            plt.rcParams['font.size'] = 20
            fig, ax3 = plt.subplots(1, 1, figsize=(10, 10))
            black_image = np.zeros((10, 10))  # Small black image
            im3 = ax3.imshow(black_image, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax3.set_title(f'{name}\nNot in both masks')
            fig.savefig(os.path.join(tboard_dir, f'{name}_importance_{suffix}.png'))
            plt.close(fig)
            continue
            
        # Process valid layers
        jac_mask = jac_masks[name].cpu().numpy()
        mag_mask = mag_masks[name].cpu().numpy()
        
        # Reshape to 2D if needed
        if len(jac_mask.shape) == 4:  # Conv layer
            c_out, c_in, h, w = jac_mask.shape
            jac_mask_2d = jac_mask.reshape(c_out, -1)
            mag_mask_2d = mag_mask.reshape(c_out, -1)
        elif len(jac_mask.shape) == 2:  # Linear layer
            jac_mask_2d = jac_mask
            mag_mask_2d = mag_mask
        else:
            continue
            
        plt.rcParams['font.size'] = 20
        fig, ax3 = plt.subplots(1, 1, figsize=(10, 10))
        
        # Calculate overlap
        overlap = jac_mask_2d + 2*mag_mask_2d
        
        # Check if there's any overlap
        if np.any(overlap == 3):
            # If there's overlap, show the normal visualization
            custom_cmap = create_custom_colormap()
            im3 = ax3.imshow(overlap, cmap=custom_cmap, aspect='auto', vmin=0, vmax=3)
            cbar = plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2, 3])
            cbar.set_ticklabels(['Neither', 'Gradient', 'Magnitude', 'Both'])
            title_suffix = ''
        else:
            # If no overlap, create black image
            black_image = np.zeros_like(overlap)
            im3 = ax3.imshow(black_image, cmap='gray', aspect='auto', vmin=0, vmax=1)
            title_suffix = '\n(No Common Parameters)'
            
        ax3.set_title(f'{name}\nOverlap{title_suffix}')
        fig.savefig(os.path.join(tboard_dir, f'{name}_importance_{suffix}.png'))
        plt.close(fig)

def visualize_weight_differences(model1_weights, model2_weights, tboard_dir, suffix1='1', suffix2='2'):
    """
    Visualize differences between two models' weights using heatmaps
    """
    for name in model1_weights.keys():
        if 'weight' in name and name in model2_weights:
            weights1 = model1_weights[name].cpu().numpy()
            weights2 = model2_weights[name].cpu().numpy()
            
            # Calculate difference
            diff = np.abs(weights1 - weights2)
            
            # Reshape to 2D if needed
            if len(weights1.shape) == 4:  # Conv layer
                c_out, c_in, h, w = weights1.shape
                weights1_2d = weights1.reshape(c_out, -1)
                weights2_2d = weights2.reshape(c_out, -1)
                diff_2d = diff.reshape(c_out, -1)
            elif len(weights1.shape) == 2:  # Linear layer
                weights1_2d = weights1
                weights2_2d = weights2
                diff_2d = diff
            else:
                continue
            
            # Create subplots
            plt.rcParams['font.size'] = 20
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            
            # Plot original weights
            im1 = ax1.imshow(weights1_2d, cmap='hot_r', aspect='auto')
            ax1.set_title(f'{name}\n{suffix1}')
            plt.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(weights2_2d, cmap='hot_r', aspect='auto')
            ax2.set_title(f'{name}\n{suffix2}')
            plt.colorbar(im2, ax=ax2)
            
            # Plot difference
            im3 = ax3.imshow(diff_2d, cmap='hot_r', aspect='auto')
            ax3.set_title(f'{name}\nDifference')
            plt.colorbar(im3, ax=ax3)
            
            # Save
            fig.savefig(os.path.join(tboard_dir, f'{name}_diff_{suffix1}vs{suffix2}.png'))
            plt.close(fig)


if __name__ == '__main__':
    args = args_parser()

    # keypair generation
    HE_method = args.he_lib
    save_dir = "save"
    ring_dim = args.ring_dim # 8192
    scale_bit = args.scale_bit # 52
    datafolder = os.path.join(save_dir, HE_method + "_" + str(ring_dim) + "_" + str(scale_bit) + "_" + args.name)
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)
    print(f"scale bit: {args.scale_bit}; datafolder name: {datafolder}")


    non_HE_time = 0
    # Start timer
    start_time = time.time()
    # Parse arguments and create/load checkpoint
    if not args.resume:
        checkpoint = {}
        checkpoint['args'] = args
    else:
        checkpoint = torch.load(f'{save_dir}/{args.name}')
        rounds = args.rounds
        iters = args.iters
        device =args.device
        args = checkpoint['args']
        args.resume = True
        args.rounds = rounds
        args.iters = iters
        args.device = device

    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        if not args.resume:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
        else:
            torch.set_rng_state(checkpoint['torch_rng_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['python_rng_state'])

    # Load datasets and splits
    if not args.resume:
        datasets = getattr(datasets, args.dataset)(args, args.dataset_args)
        splits = get_splits(datasets, args.num_clients, args.iid, args.balance)
        datasets_actual = {}
        for dataset_type in splits:
            if splits[dataset_type] is not None:
                idxs = []
                for client_id in splits[dataset_type].idxs:
                    idxs += splits[dataset_type].idxs[client_id]

                if args.dataset == "tiny_imagenet":
                    datasets_actual[dataset_type] = TinyImageNetSubset(datasets[dataset_type], idxs)
                else:
                    datasets_actual[dataset_type] = Subset(datasets[dataset_type], idxs)
            else:
                datasets_actual[dataset_type] = None
        checkpoint['splits'] = splits
        checkpoint['datasets_actual'] = datasets_actual
    else:
        splits = checkpoint['splits']
        datasets_actual = checkpoint['datasets_actual']
    acc_types = ['train', 'test'] if datasets_actual['valid'] is None else ['train', 'valid']

    # Load model
    num_classes = len(datasets_actual['train'].classes)
    num_channels = datasets_actual['train'][0][0].shape[0]
    model = getattr(models, args.model)(num_classes, num_channels, args.model_args).to(args.device)
    model_param = deepcopy(model.state_dict())


    # Load optimizer and scheduler
    optim = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    sched = getattr(schedulers, args.sched)(optim, args.sched_args)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])
        sched.load_state_dict(checkpoint['sched_state_dict'])

    # Create clients
    if not args.resume:
        clients = []
        for client_id in range(args.num_clients):
            client_idxs = {dataset_type: splits[dataset_type].idxs[client_id] if splits[dataset_type] is not None else None for dataset_type in splits}
            clients.append(Client(args=args, datasets=datasets, idxs=client_idxs))
        checkpoint['clients'] = clients
    else:
        clients = checkpoint['clients']

    # Set client sampling probabilities
    if args.vc_size is not None:
        # Proportional to the number of examples (FedVC)
        p_clients = np.array([len(client.loaders['train'].dataset) for client in clients])
        p_clients = p_clients / p_clients.sum()
    else:
        # Uniform
        p_clients = None

    # Determine number of clients to sample per round
    m = max(int(args.frac_clients * args.num_clients), 1)

    # Print experiment summary
    summary = exp_details(args, model, datasets_actual, splits)
    print('\n' + summary)

    init_end_time = time.time()

    # Train server model
    if not args.resume:
        last_round = -1
        iter = 0
        v = None
    else:
        last_round = checkpoint['last_round']
        iter = checkpoint['iter']
        v = checkpoint['v']

    
    # load_smap_dir_jac = "/media/renyi/Data/RenYi/invertinggradients-master_20241113/cifar100_models/newLeNet_jac_0.5/average_global_map"
    # load_smap_dir_mag = "/media/renyi/Data/RenYi/invertinggradients-master_20241113/cifar100_models/newLeNet_mag_0.5/average_global_map_mag"
    load_smap_dir_jac = "/media/renyi/Data/RenYi/invertinggradients-master_20241113/cifar100_models/resnet18_mag/average_global_map_mag"
    load_smap_dir_mag = "/media/renyi/Data/RenYi/invertinggradients-master_20241113/cifar100_models/resnet18_mag/average_global_map_mag"
    
    average_global_map_jac = torch.load(load_smap_dir_jac)
    average_global_map_mag = torch.load(load_smap_dir_mag)
    print(f"Smap {load_smap_dir_jac} loaded")

    overlap_percentage_dict = {}

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for draw_ratio in ratios:
        folder = f"./comparison_{draw_ratio}"
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Server decides mask based on global sensitivity map and ratio
        top_params_jac, s_mask_jac = smap_utils.sort_and_get_top_params_globally(average_global_map_jac, draw_ratio)
        top_params_mag, s_mask_mag = smap_utils.sort_and_get_top_params_globally(average_global_map_mag, draw_ratio)

        compared_result = compare_weight_locations(top_params_jac, top_params_mag) 
        overlap_percentage_dict[draw_ratio] = compared_result["overlap_percentage"]

        print(draw_ratio)
        # print(compared_result)
        print("=====\n\n")

        visualize_importance_comparison(average_global_map_jac, top_params_jac, top_params_mag, folder, suffix=f'visualize_importance_comparison_{draw_ratio}')
    
    print(f"in the end, overlap_percentage_dict: {overlap_percentage_dict}")

    stop =1
