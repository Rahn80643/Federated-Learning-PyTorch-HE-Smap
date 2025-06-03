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

'''
1. During initilization, keypair is generated for HE and the model is encrypted by a trusted third party or trusted server 
2. Sensitivity maps are calculated by a initilized and encrypted model in the clients' side. Next, the server aggregates the maps and generates mask for encryption
3. The clients perform local training with sensitivity maps, and return trained-encrypted model and trained-unencrypted model
4. The server aggregates encrypted and unencrypted models separately and send to the clients for the next training round

'''

if __name__ == '__main__':
    args = args_parser()

    # keypair generation
    HE_method = args.he_lib
    save_dir = "save"
    ring_dim = args.ring_dim 
    scale_bit = args.scale_bit 
    datafolder = os.path.join(save_dir, HE_method + "_" + str(ring_dim) + "_" + str(scale_bit) + "_" + args.name)
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)
    print(f"scale bit: {args.scale_bit}; datafolder name: {datafolder}")

    if HE_method == "OpenFHE_CKKS":
        pkContext, skContext, HE_context, keygen_time, keysize_result = keygen.key_gen_OpenFHE_CKKS(ring_dim, scale_bit, datafolder)
            
    elif HE_method == "TenSeal_CKKS" or HE_method == "TenSeal_CKKS_without_flatten": 
        pkContext, skContext, HE_context, keygen_time, keysize_result = keygen.key_gen_TenSeal_CKKS(ring_dim, scale_bit, datafolder)
            
    elif HE_method == "Pyfhel_CKKS":
        pkContext, skContext, HE_context, keygen_time, keysize_result = keygen.key_gen_Pyfhel_CKKS(ring_dim, scale_bit, datafolder)

    else:
        print("Unsupported library")

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

    if args.resume: # if resume training the model
        if HE_method == "Pyfhel_CKKS":
            model_enc_global = checkpoint['model_enc']            

    else: # if traning from scratch     
        model_enc_global, model_layer_size_enc, _, _ = he_utils.encrypt_model_new(HE_method, model, HE_context, pkContext, skContext)        

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

    # Log experiment summary, client distributions, example images
    if not args.no_log:
        logger = SummaryWriter(f'runs/{args.name}')
        if not args.resume:
            logger.add_text('Experiment summary', re.sub('^', '    ', re.sub('\n', '\n    ', summary)))

            splits_fig = get_splits_fig(splits, args.iid, args.balance)
            logger.add_figure('Splits', splits_fig)
            
            if args.dataset == "tiny_imagenet":
                datasets_fig = get_datasets_fig_tiny_imagenet(datasets_actual, args.train_bs)
            else:
                datasets_fig = get_datasets_fig(datasets_actual, args.train_bs)
            logger.add_figure('Datasets', datasets_fig)

            input_size = (1,) + tuple(datasets_actual['train'][0][0].shape)
            fake_input = torch.zeros(input_size).to(args.device)
            if "ViT" not in args.model: # TODO: ViT can't be used for add_graph()
                logger.add_graph(model, fake_input)
    else:
        logger = None

    if not args.resume:
        # Compute initial average accuracies
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
        acc_avg_best = acc_avg[acc_types[1]]

        # Print and log initial stats
        if not args.quiet:
            print('Training:')
            print('    Round: 0' + (f'/{args.rounds}' if args.iters is None else ''))
        loss_avg, lr = torch.nan, torch.nan
        printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, 0, 0, args.iters)
    else:
        acc_avg_best = checkpoint['acc_avg_best']

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

    # Generate sensitivy maps in clients' side
    torch.save(model.state_dict(), f"{datafolder}/model_init_dict")
    if not args.load_map: # not loading previously created maps, generate maps
        if args.s_method == "magnitude":
            # generate maps based on magnitude
            s_map_gen_t_start = time.time()
            # Generate sensitivity maps for ALL clients (paper sectin 2.3)
            clients_model_s_maps_mag, clients_model_s_exec_time_dicts_mag, clients_model_s_s_map_sums_mag, clients_model_s_map_avgs_mag = {}, {}, {}, {}
            clients_model_s_maps_list_mag = []
            client_ids = range(args.num_clients)
            for i, client_id in enumerate(client_ids): 
                if not args.quiet: print(f'[Sensitivity map] Client: {client_id} ({i+1}/{m})')

                client_enc_model = deepcopy(model_enc_global) # make sure clients get encrypted model, but it can't be deep copied if using public key
                optim.__setstate__({'state': defaultdict(dict)})

                model_s_map_mag, exec_time_dict_mag, s_maps_mag_avg = \
                    clients[client_id].gen_map_mag(model=model, model_enc=client_enc_model, model_param=model_param, model_layer_size_enc=model_layer_size_enc, optim=optim, device=args.device, HE_context=HE_context, pub_context=pkContext, priv_context=skContext, HE_method=HE_method, args=args)

                clients_model_s_maps_mag[client_id] = model_s_map_mag
                clients_model_s_maps_list_mag.append(model_s_map_mag)
                clients_model_s_exec_time_dicts_mag[client_id] = exec_time_dict_mag
                clients_model_s_map_avgs_mag[client_id] = s_maps_mag_avg

            s_map_gen_t_end = time.time()
            s_map_gen_time = s_map_gen_t_end - s_map_gen_t_start 
            
            # # Server aggregates sensitivity map
            s_map_agg_t_start = time.time()
            average_global_map_mag = average_s_map(clients_model_s_maps_list_mag)
            s_map_agg_t_end = time.time()
            s_map_agg_time = s_map_agg_t_end - s_map_agg_t_start 

            # save individual client's map and aggregated map
            for i, client_id in enumerate(client_ids): 
                smap_utils.save_visualized_layer_weights(clients_model_s_maps_mag[client_id], datafolder, "mag_"+str(client_id))

            smap_utils.save_visualized_layer_weights(average_global_map_mag, datafolder, "server_global_mag")
            torch.save(clients_model_s_maps_mag, f"{datafolder}/clients_model_s_maps_mag")
            torch.save(clients_model_s_exec_time_dicts_mag, f"{datafolder}/clients_model_s_exec_time_dicts_mag")
            torch.save(clients_model_s_s_map_sums_mag, f"{datafolder}/clients_model_s_s_map_sums_mag")
            torch.save(clients_model_s_map_avgs_mag, f"{datafolder}/clients_model_s_map_avgs_mag")
            torch.save(average_global_map_mag, f"{datafolder}/average_global_map_mag")
            average_global_map = average_global_map_mag
            # Finish creating smaps
        
        elif args.s_method == "jacobian": # Sensitivity map based on FedML-HE
            # generate maps based on second derivatives
            s_map_gen_t_start = time.time()
            # Generate sensitivity maps for ALL clients (paper sectin 2.3)
            clients_model_s_maps, clients_model_s_exec_time_dicts, clients_model_s_s_map_sums, clients_model_s_map_avgs = {}, {}, {}, {}
            clients_model_s_maps_list = []
            client_ids = range(args.num_clients)
            for i, client_id in enumerate(client_ids): 
                if not args.quiet: print(f'[Sensitivity map] Client: {client_id} ({i+1}/{m})')

                client_enc_model = deepcopy(model_enc_global) # make sure clients get encrypted model, but it can't be deep copied if using public key
                optim.__setstate__({'state': defaultdict(dict)})

                model_s_map, exec_time_dict, s_map_sum, map_avg = \
                    clients[client_id].gen_map_grad(model=model, model_enc=client_enc_model, model_param=model_param, model_layer_size_enc=model_layer_size_enc, optim=optim, device=args.device, HE_context=HE_context, pub_context=pkContext, priv_context=skContext, HE_method=HE_method, args=args)

                clients_model_s_maps[client_id] = model_s_map
                clients_model_s_maps_list.append(model_s_map)
                clients_model_s_exec_time_dicts[client_id] = exec_time_dict
                clients_model_s_s_map_sums[client_id] = s_map_sum
                clients_model_s_map_avgs[client_id] = map_avg

            s_map_gen_t_end = time.time()
            s_map_gen_time = s_map_gen_t_end - s_map_gen_t_start 
            
            # # Server aggregates sensitivity map
            s_map_agg_t_start = time.time()
            average_global_map = average_s_map(clients_model_s_maps_list)
            s_map_agg_t_end = time.time()
            s_map_agg_time = s_map_agg_t_end - s_map_agg_t_start 

            # save individual client's map and aggregated map
            for i, client_id in enumerate(client_ids): 
                smap_utils.save_visualized_layer_weights(clients_model_s_maps[client_id], datafolder, "jac_"+str(client_id))

            smap_utils.save_visualized_layer_weights(average_global_map, datafolder, "server_global_jacobian")
            torch.save(clients_model_s_maps, f"{datafolder}/clients_model_s_maps")
            torch.save(clients_model_s_exec_time_dicts, f"{datafolder}/clients_model_s_exec_time_dicts")
            torch.save(clients_model_s_s_map_sums, f"{datafolder}/clients_model_s_s_map_sums")
            torch.save(clients_model_s_map_avgs, f"{datafolder}/clients_model_s_map_avgs")
            torch.save(average_global_map, f"{datafolder}/average_global_map")
        # Finish creating smaps

    else: # if load previously created smaps
        s_map_gen_time = -1
        average_global_map = torch.load(args.load_smap_dir)
        s_map_agg_time = -1
        torch.save(average_global_map, f"{datafolder}/average_global_map")

    # Server decides mask based on global sensitivity map and encryption ratio (s_ratio)
    # s_mask indicates which parameters should be encrypted
    mask_gen_t_end = time.time()
    top_params, s_mask = smap_utils.sort_and_get_top_params_globally(average_global_map, args.s_ratio)
    mask_gen_t_end = time.time()
    mask_gen_time = mask_gen_t_end - mask_gen_t_end 

    print("Sensitivity map generated!")

    # Train clients
    c_part_enc_layer_size = {}
    mask_indices = {}
    model_ptxt_avg = deepcopy(model.state_dict()) # first round, model_ptxt_avg is from global model

    local_train_time_dicts, HE_enc_time_dicts, non_HE_flat_time_dicts, HE_dec_time_dicts, \
        non_HE_reshape_ptxt_time_dicts, non_HE_reshape_ctxt_time_dicts, HE_ctxt_agg_time_dict, non_HE_time_dict =\
            {}, {}, {}, {}, {}, {}, {}, {}
    
    nonHE_ptxt_agg_time_dict = {}
    non_HE_recover_model_time_dicts, non_HE_mask_model_time_dicts, non_HE_select_flat_time_dicts = {}, {}, {}
    for round in range(last_round + 1, args.rounds):
        if not args.quiet:
            print(f'    Round: {round+1}' + (f'/{args.rounds}' if args.iters is None else ''))

        # Sample clients
        sample_client_t_start = time.time()
        client_ids = np.random.choice(range(args.num_clients), m, replace=False, p=p_clients)
        sample_client_t_end = time.time()
        sample_client_time = sample_client_t_end - sample_client_t_start
        non_HE_time += sample_client_time

        # Train client models
        local_train_time_dict, HE_enc_time_dict, non_HE_flat_time_dict, \
            HE_dec_time_dict, non_HE_reshape_ptxt_time_dict, non_HE_reshape_ctxt_time_dict = \
              {}, {}, {}, {}, {}, {}
        non_HE_recover_model_time_dict, non_HE_mask_model_time_dict, non_HE_select_flat_time_dict = {}, {}, {}
        updates, num_examples, max_iters, loss_tot = [], [], 0, 0.
        client_models = [] # plaintext for validate HE results
        c_ptxt_models_dicts = []
        enc_models = [] 
        layer_sizes = [] 
        for i, client_id in enumerate(client_ids):
            if not args.quiet: print(f'        Client: {client_id} ({i+1}/{m})')

            client_enc_model = deepcopy(model_enc_global) # make sure clients get encrypted model, but it can't be deep copied if using public key if using OpenFHE
            client_ptxt_model_dict = deepcopy(model_ptxt_avg) # clients receive the global model

            optim.__setstate__({'state': defaultdict(dict)})

            # attention: 
            ## client_update: not used, it's the difference between local model and global model
            ## client_model_dict_comparison: for validation
            ## c_part_enc_model_dict: for aggregate encrypted models
            ## c_part_ptxt_model_dict: for aggregate plaintext models

            client_update, client_model_dict_comparison, \
                c_part_enc_model_dict, c_part_ptxt_model_dict, c_part_enc_layer_size, mask_indices, \
                client_num_examples, client_num_iters, client_loss, local_train_time, \
                HE_enc_time, non_HE_flat_time, HE_dec_time, non_HE_reshape_ctxt_time, \
                non_HE_recover_model_time, non_HE_reshape_ptxt_time, non_HE_mask_model_time, non_HE_select_flat_time \
                = clients[client_id].train_with_smaps(model=model, model_enc_dict=client_enc_model, model_ptxt_dict=client_ptxt_model_dict, model_param=model_param, \
                                            model_layer_size_enc=c_part_enc_layer_size, optim=optim, \
                                            device=args.device, HE_context=HE_context, pub_context=pkContext, priv_context=skContext, \
                                            HE_method=HE_method, args=args, \
                                            n_round=round, s_mask=s_mask, mask_indices=mask_indices, top_params=top_params)
            
                        
            if round % 10 == 0:
                torch.save(c_part_ptxt_model_dict, f'{datafolder}/{args.name}-round-{round}-client-{client_id}.model') # validate client ptxt size
            
            after_training_t_start = time.time()
            if client_num_iters > max_iters: max_iters = client_num_iters

            if client_update is not None:
                updates.append(deepcopy(client_update))
                loss_tot += client_loss * client_num_examples
                num_examples.append(client_num_examples)

            if client_model_dict_comparison is not None:
                client_models.append(deepcopy(client_model_dict_comparison))

            if c_part_enc_model_dict is not None: 
                # deepcopy error for pyfhel: CKKS scheme requires a list of prime sizes (qi_sizes) or primes (qi) to be set
                enc_models.append(deepcopy(c_part_enc_model_dict))

            if c_part_ptxt_model_dict is not None:
                c_ptxt_models_dicts.append(deepcopy(c_part_ptxt_model_dict))
            
            if model_layer_size_enc is not None:
                layer_sizes.append(deepcopy(model_layer_size_enc))
                # layer_sizes = model_layer_size_enc

            local_train_time_dict[client_id] = local_train_time
            HE_enc_time_dict[client_id] = HE_enc_time
            non_HE_flat_time_dict[client_id] = non_HE_flat_time
            HE_dec_time_dict[client_id] = HE_dec_time
            non_HE_recover_model_time_dict[client_id] = non_HE_recover_model_time
            non_HE_mask_model_time_dict[client_id] = non_HE_mask_model_time
            non_HE_select_flat_time_dict[client_id] = non_HE_select_flat_time

            non_HE_reshape_ptxt_time_dict[client_id] = non_HE_reshape_ptxt_time
            non_HE_reshape_ctxt_time_dict[client_id] = non_HE_reshape_ctxt_time
            after_training_t_end = time.time()

            after_training_time = after_training_t_end - after_training_t_start
            non_HE_time += after_training_time

        # finish training each client
        
        before_agg_t_start = time.time()
        local_train_time_dicts[round] = local_train_time_dict
        HE_enc_time_dicts[round] = HE_enc_time_dict
        non_HE_flat_time_dicts[round] = non_HE_flat_time_dict
        HE_dec_time_dicts[round] = HE_dec_time_dict
        non_HE_reshape_ptxt_time_dicts[round] = non_HE_reshape_ptxt_time_dict
        non_HE_reshape_ctxt_time_dicts[round] = non_HE_reshape_ctxt_time_dict
        
        non_HE_recover_model_time_dicts[round] = non_HE_recover_model_time_dict
        non_HE_mask_model_time_dicts[round] = non_HE_mask_model_time_dict
        non_HE_select_flat_time_dicts[round] = non_HE_select_flat_time_dict

        iter += max_iters
        lr = optim.param_groups[0]['lr']
        before_agg_t_end = time.time()

        before_agg_time = before_agg_t_end - before_agg_t_start
        non_HE_time += before_agg_time # not used, because only keep time record

        nonHE_ptxt_agg_time = -1
        if len(c_ptxt_models_dicts) > 0: # perform aggregation
            # Update server model
            # since non-iid only affects the distribution of classes in dataset, all clients still get the same number of data points; average_updates == average_updates_mod 
            # update_avg = average_updates(updates, num_examples)
            # update_avg = average_updates_mod(updates)
            # model_avg = average_updates_mod(client_models)

            if args.s_ratio < 1.0: # if not fully encryption
                nonHE_ptxt_agg_t_start = time.time()
                model_ptxt_avg = average_updates_mod(c_ptxt_models_dicts)
                nonHE_ptxt_agg_t_end = time.time()
                nonHE_ptxt_agg_time = nonHE_ptxt_agg_t_end - nonHE_ptxt_agg_t_start
            else:
                model_ptxt_avg = {}
            # Finish aggregating plaintext 

        HE_ctxt_agg_time = -1
        if len(enc_models) > 0 and args.s_ratio > 0.0: # perform aggregation
            # compare model.state_dict() and model_avg
            if HE_method == "OpenFHE_CKKS":
                update_avg_HE, HE_ctxt_agg_time = he_utils.average_updates_HE_OpenFHE_CKKS(enc_models, HE_context) 

            elif HE_method == "TenSeal_CKKS" or HE_method == "TenSeal_CKKS_without_flatten":
                update_avg_HE, HE_ctxt_agg_time = he_utils.average_updates_HE_TenSeal_CKKS(enc_models)  
                    
            elif HE_method == "Pyfhel_CKKS":
                update_avg_HE, HE_ctxt_agg_time = he_utils.average_updates_HE_Pyfhel_CKKS(enc_models)
        # Finish aggregating ciphertexts

        
        HE_ctxt_agg_time_dict[round] = HE_ctxt_agg_time 
        nonHE_ptxt_agg_time_dict[round] = nonHE_ptxt_agg_time
        non_HE_time_dict[round] = non_HE_time

        # Compute round average loss and accuracies
        # Accuracy is from model_dec (model_dec + copy from state_dict())
        # decrypt ctxt model for measure accuracy in this round
        if args.s_ratio > 0.0:
            model_dec, _, _ \
                = he_utils.partial_decrypt_model_separate(HE_method, update_avg_HE, c_part_enc_layer_size, \
                                                            HE_context, skContext, model.state_dict(), mask_indices, args)
        if args.s_ratio < 1.0: # plaintext exists, recover from 1D to original shape
            recover_model_ptxt_avg, _ = smap_utils.recover_ptxt_shape(model, model_ptxt_avg, s_mask)

            for key in model_ptxt_avg:
                if key not in recover_model_ptxt_avg:
                    recover_model_ptxt_avg[key] = model_ptxt_avg[key]

        # combine ptxt and ctxt together based on differernt s_ratio
        if args.s_ratio > 0.0 and args.s_ratio < 1.0: 
            validation_model_dec = deepcopy(model_dec) # todo: DEL
            for key in recover_model_ptxt_avg.keys(): # each layer
                if key in model_dec:
                    before = model_dec[key]
                    # model_dec[key] = torch.add(model_dec[key], model_ptxt_avg[key]) 
                    model_dec[key] = torch.add(model_dec[key], recover_model_ptxt_avg[key]) 
                    after = model_dec[key]
                    stop =1 # attention: check if plaintext and decrypted models are correctly elementwise-added
                else:
                    model_dec[key] = recover_model_ptxt_avg[key]

            model.load_state_dict(model_dec) # copy decrypted weight to the global model, and for measuring accuracy
        
        elif args.s_ratio == 0.0: # pure plaintext
            model.load_state_dict(recover_model_ptxt_avg) # copy decrypted weight to the global model, and for measuring accuracy

        elif args.s_ratio == 1.0: # pure ciphertext
            model.load_state_dict(model_dec) # copy decrypted weight to the global model, and for measuring accuracy

        if round % args.server_stats_every == 0:
            loss_avg = loss_tot / sum(num_examples)
            acc_avg = get_acc_avg(acc_types, clients, model, args.device)

            if acc_avg[acc_types[1]] > acc_avg_best:
                acc_avg_best = acc_avg[acc_types[1]]
        # training round ends
        
        # Additional step: save models and check accuracy
        # Save checkpoint
        # global_model_dec: for measuring the size of plaintext model
        checkpoint["HE_method"] = HE_method

        if HE_method == "Pyfhel_CKKS" and args.s_ratio > 0.0:
            checkpoint['model_enc'] = update_avg_HE 
            # update_avg_HE.save(os.path.join(datafolder, "model_enc"))
            if (round+1) % 10 == 0:
                torch.save(update_avg_HE, f'{datafolder}/model_enc-round-{round}.model')
        #         global_model_dec, _, _ = he_utils.decrypt_model_separate(HE_method, update_avg_HE, model_layer_size_enc, HE_context, skContext, model.state_dict(), args)
        
        elif HE_method == "TenSeal_CKKS" or HE_method == "TenSeal_CKKS_without_flatten" and args.s_ratio > 0.0:
            # https://github.com/OpenMined/TenSEAL/blob/main/tenseal/__init__.py
            # similar to OpenFHE, serialization only supports on individual ciphertext
            model_enc_dir = os.path.join(datafolder, "model_enc")
            if not os.path.exists(model_enc_dir):
                os.mkdir(model_enc_dir)

            for key in update_avg_HE:
                enc_layer = update_avg_HE[key]

                enc_layer_data = enc_layer.serialize()
                with open(os.path.join(model_enc_dir, key), "wb") as f:
                    # print(f"writing key: {key}")
                    f.write(enc_layer_data)


        elif HE_method == "OpenFHE_CKKS" and args.s_ratio > 0.0:
            # since SerializeToFile in OpenFHE only supports serialize each ciphertext, save all of them individually along with the key
            # iterate through each layer in dict "update_avg_HE"
            model_enc_dir = os.path.join(datafolder, "model_enc")
            if not os.path.exists(model_enc_dir):
                os.mkdir(model_enc_dir)
            
            for key in update_avg_HE:
                enc_layer = update_avg_HE[key]

                if not SerializeToFile(os.path.join(model_enc_dir, key), enc_layer, BINARY):
                    raise Exception("Error writing serialization of model_enc to model_enc")
                # print("model_enc for OpenFHE has been serialized.")
    
        if args.s_ratio > 0.0:
            # for validate encrypted content, not for training
            model_dec_ctxt_dict, _, _ = \
            he_utils.partial_decrypt_model_separate(HE_method, update_avg_HE, model_layer_size_enc, \
                                                    HE_context, skContext, model.state_dict(), mask_indices, args)# TODO: DEL, only validate if encryption is correct

            # save aggregated models for the next training round
            model_enc_global = deepcopy(update_avg_HE) 
        

        # for validating final accuracy
        if args.s_ratio > 0.0:
            model.load_state_dict(model_dec) # model_dec is alread the sum of ptxt and decrypted ctxt
        elif args.s_ratio == 0.0:
            model.load_state_dict(recover_model_ptxt_avg)

        if (round+1) % 10 == 0:
            torch.save(model.state_dict(), f'{datafolder}/{args.name}-round-{round}.model')
        model_size_ptxt = measure_model_size(model)

        checkpoint["keygen_time"] = keygen_time
        checkpoint["s_map_gen_time"] = s_map_gen_time
        checkpoint["s_map_agg_time"] = s_map_agg_time
        checkpoint["mask_gen_time"] = mask_gen_time

        checkpoint["model_size_ptxt"] = model_size_ptxt
        checkpoint["non_HE_recover_model_times"] = non_HE_recover_model_time_dict
        checkpoint["non_HE_mask_model_times"] = non_HE_mask_model_time_dict
        checkpoint["non_HE_select_flat_times"] = non_HE_select_flat_time_dict
        checkpoint["clients_enc_time"] = HE_enc_time_dict
        checkpoint["clients_non_HE_flat_time"] = non_HE_flat_time_dict
        checkpoint["clients_HE_dec_time"] = HE_dec_time_dict
        checkpoint["clients_non_HE_reshape_ptxt_time"] = non_HE_reshape_ptxt_time_dict
        checkpoint["clients_train_time"] = local_train_time_dict

        checkpoint['model_layer_size'] = model_layer_size_enc
        checkpoint['HE_key_folder'] = datafolder
        # checkpoint["HE_context"] = HE_context # TODO: del: can't get keys if saving this way

        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['optim_state_dict'] = optim.state_dict()
        checkpoint['sched_state_dict'] = sched.state_dict()
        checkpoint['last_round'] = round
        checkpoint['iter'] = iter
        checkpoint['v'] = v
        checkpoint['acc_avg_best'] = acc_avg_best
        checkpoint['torch_rng_state'] = torch.get_rng_state()
        checkpoint['numpy_rng_state'] = np.random.get_state()
        checkpoint['python_rng_state'] = random.getstate()

        # OpenFHE issue: can't pickle openfhe.Ciphertext objects, can't validate load models
        # torch.save(checkpoint, f'save/{args.name}-round-{round}')
        if (round+1) % 10 == 0:
            torch.save(checkpoint, f'{datafolder}/{args.name}-round-{round}')
        
        # Print and log round stats
        if round % args.server_stats_every == 0:
            printlog_stats(args.quiet, logger, loss_avg, acc_avg, acc_types, lr, round+1, iter, args.iters)

        # Stop training if the desired number of iterations has been reached
        if args.iters is not None and iter >= args.iters: break

        # Step scheduler
        if type(sched) == schedulers.plateau_loss:
            sched.step(loss_avg)
        else:
            sched.step()

    # training round ends
    
    # keep record of time for all clients in all rounds
    checkpoint["non_HE_times"] = non_HE_time_dict
    checkpoint["server_HE_ctxt_agg_times"] = HE_ctxt_agg_time_dict
    checkpoint["server_nonHE_ptxt_agg_times"] = nonHE_ptxt_agg_time_dict
    checkpoint["local_train_time_dicts"] = local_train_time_dicts
    checkpoint["HE_enc_time_dicts"] = HE_enc_time_dicts
    checkpoint["non_HE_flat_time_dicts"] = non_HE_flat_time_dicts
    checkpoint["HE_dec_time_dicts"] = HE_dec_time_dicts
    checkpoint["non_HE_recover_model_time_dicts"] = non_HE_recover_model_time_dicts
    checkpoint["non_HE_reshape_ptxt_time_dicts"] = non_HE_reshape_ptxt_time_dicts
    checkpoint["non_HE_reshape_ctxt_time_dicts"] = non_HE_reshape_ctxt_time_dicts
    checkpoint["non_HE_mask_model_time_dicts"] = non_HE_mask_model_time_dicts
    checkpoint["non_HE_select_flat_time_dicts"] = non_HE_select_flat_time_dicts
    
    # torch.save(checkpoint, f'save/{args.name}-final')
    torch.save(checkpoint, f'{datafolder}/{args.name}-final')
    torch.save(model.state_dict(), f'{datafolder}/{args.name}-final.model')
    
    if HE_method == "Pyfhel_CKKS":
        torch.save(update_avg_HE, f'{datafolder}/model_enc-final.model')
    train_end_time = time.time()

    # Compute final average test accuracy
    acc_avg = get_acc_avg(['test'], clients, model, args.device)

    test_end_time = time.time()

    # Print and log test results
    print('\nResults:')
    print(f'    Average test accuracy: {acc_avg["test"]:.3%}')
    print(f'    Train time: {timedelta(seconds=int(train_end_time-init_end_time))}')
    print(f'    Total time: {timedelta(seconds=int(time.time()-start_time))}')

    if logger is not None: logger.close()