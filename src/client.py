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

from copy import deepcopy
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils import parameters_to_vector
import smap_utils as smap_utils

from utils import inference

# for HE
import tenseal as ts
import openfhe
from openfhe import *
from Pyfhel import Pyfhel
import he_utils as he_utils 

class Client(object):
    def __init__(self, args, datasets, idxs):
        self.args = args

        # Create dataloaders
        self.train_bs = self.args.train_bs if self.args.train_bs > 0 else len(idxs['train'])
        self.loaders = {}
        self.loaders['train'] = DataLoader(Subset(datasets['train'], idxs['train']), batch_size=self.train_bs, shuffle=True) if len(idxs['train']) > 0 else None
        self.loaders['valid'] = DataLoader(Subset(datasets['valid'], idxs['valid']), batch_size=args.test_bs, shuffle=False) if idxs['valid'] is not None and len(idxs['valid']) > 0 else None
        self.loaders['test'] = DataLoader(Subset(datasets['test'], idxs['test']), batch_size=args.test_bs, shuffle=False) if len(idxs['test']) > 0 else None

        # Set criterion
        if args.fedir:
            # Importance Reweighting (FedIR)
            labels = set(datasets['train'].targets)
            p = torch.tensor([(torch.tensor(datasets['train'].targets) == label).sum() for label in labels]) / len(datasets['train'].targets)
            q = torch.tensor([(torch.tensor(datasets['train'].targets)[idxs['train']] == label).sum() for label in labels]) / len(torch.tensor(datasets['train'].targets)[idxs['train']])
            weight = p/q
        else:
            # No Importance Reweighting
            weight = None
        self.criterion = CrossEntropyLoss(weight=weight)
        self.criterion_smap_magnitude = CrossEntropyLoss(reduction='none') # obtain all loss items

    # generate sensitivity map based on jacobian of graidents
    def gen_map_grad(self, model, model_enc, model_param, model_layer_size_enc, optim, device, HE_context, pub_context, priv_context, HE_method, args):
        # Decrypt model
        model_dict, dec_time, reshape_time = he_utils.decrypt_model_separate(HE_method, model_enc, model_layer_size_enc, HE_context, priv_context, model_param, args)# TODO: DEL, only validate if encryption is correct
        model.load_state_dict(model_dict) # get decrypted weight

        # Create training loader
        if self.args.vc_size is not None:
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.vc_size:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.vc_size]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.vc_size,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']

        # one epoch (iterate through all batches of data) to get the gradient
        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)
        
        model.to(device)
        model.train()
        self.criterion.to(device)

        s_map_sum = deepcopy(model) # for storing sensitivity maps
        s_map_sum.to(device)
        # make sure all parameters for s_map_sum are 0 instead of randomly initialized
        for name, param in s_map_sum.named_parameters():
            torch.nn.init.zeros_(s_map_sum.state_dict()[name])

        # # Verify that all parameters are initialized to 0
        # for name, param in s_map_sum.named_parameters():
        #     print(f"{name}: {param.unique()}; {s_map_sum.state_dict()[name].shape}")
            
        # s_maps_all = [] # = [{} for _ in range(len(train_loader.dataset))] # TODO: verify if consumes lots of memory, for all data points
        s_maps_layer_time = [{} for _ in range(len(train_loader.dataset))] # for saving execution time of smap per batch; TODO: mod len(train_loader.dataset) as number of batches
        jm_batch_times = []
        jm_layer_second_der_times = []

        smap_all_start_t = time.time()
        # num_images_jacob: for logging progress of batch        
        num_images_jacob, s_maps_jacob, s_maps_layer_time, jm_layer_second_der_times, jm_batch_times, smap_jm_time = \
            smap_utils.calculate_smap_jacob(train_loader, args.train_bs, model, self.criterion, device)


        print("calculating average map of the dataset")
        map_avg_jacob_orig, map_sum_jacob, smap_avg_time = \
            smap_utils.avg_batch_maps(s_maps_jacob, model, num_images_jacob)

        model_s_map_jacob = deepcopy(s_map_sum) # s_map_sum: contain all 0s
        model_s_map_jacob.load_state_dict(map_avg_jacob_orig, strict=False)

        smap_all_end_t = time.time()
        smap_all_time = smap_all_end_t - smap_all_start_t
        print(f"[Jacobian] time taken for sensitivity map: {smap_all_time}")
        print(f"    [Jacobian] time taken for calculate individual jm: {smap_jm_time}")
        print(f"    [Jacobian] time taken for calculate average sensitivity map: {smap_avg_time}")

        exec_time_dict_jacob = {}
        exec_time_dict_jacob["smap_all_time"] = smap_all_time
        exec_time_dict_jacob["smap_jm_time"] = smap_jm_time
        exec_time_dict_jacob["smap_calc_avg_time"] = smap_avg_time
        exec_time_dict_jacob["jm_layer_second_der_times"] = jm_layer_second_der_times
        exec_time_dict_jacob["s_maps_layer_time"] = s_maps_layer_time
        exec_time_dict_jacob["jm_batch_times"] = jm_batch_times

        # TODO: encrypt smap


        return model_s_map_jacob, exec_time_dict_jacob, s_map_sum, map_avg_jacob_orig
    # end of sensitivity map

    # generate sensitivity map based on magnitude
    def gen_map_mag(self, model, model_enc, model_param, model_layer_size_enc, optim, device, HE_context, pub_context, priv_context, HE_method, args):
        # Decrypt model
        model_dict, dec_time, reshape_time = he_utils.decrypt_model_separate(HE_method, model_enc, model_layer_size_enc, HE_context, priv_context, model_param, args)# TODO: DEL, only validate if encryption is correct
        model.load_state_dict(model_dict) # get decrypted weight

        # Create training loader
        if self.args.vc_size is not None:
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.vc_size:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.vc_size]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.vc_size,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']

        # one epoch (iterate through all batches of data) to get the gradient
        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)
        
        model.to(device)
        model.train()
        self.criterion_smap_magnitude.to(device)

        s_map_sum = deepcopy(model) # for storing sensitivity maps
        s_map_sum.to(device)
        # make sure all parameters for s_map_sum are 0 instead of randomly initialized
        for name, param in s_map_sum.named_parameters():
            torch.nn.init.zeros_(s_map_sum.state_dict()[name])

        # # Verify that all parameters are initialized to 0
        # for name, param in s_map_sum.named_parameters():
        #     print(f"{name}: {param.unique()}; {s_map_sum.state_dict()[name].shape}")
            

        smap_all_start_t = time.time()
        # num_images_jacob: for logging progress of batch
        s_maps_mag_avg, smap_batch_times, smap_time = \
            smap_utils.calculate_smap_magnitude(train_loader, args.train_bs, model, self.criterion_smap_magnitude, device)


        model_s_map_mag = deepcopy(s_map_sum) # s_map_sum: contain all 0s
        model_s_map_mag.load_state_dict(s_maps_mag_avg, strict=False)

        smap_all_end_t = time.time()
        smap_all_time = smap_all_end_t - smap_all_start_t
        print(f"[Magnitude] time taken for sensitivity map: {smap_all_time}")
        print(f"    [Magnitude] time taken for: {smap_time}") # DEL

        exec_time_dict_mag = {}
        exec_time_dict_mag["smap_all_time"] = smap_all_time
        exec_time_dict_mag["smap_batch_times"] = smap_batch_times

        return model_s_map_mag, exec_time_dict_mag, s_maps_mag_avg
    # end of sensitivity map
  
        

    # Training client's model with homomorphic encryption and partial encryption; the parameter to be encrypted is controlled by s_mask
    def train_with_smaps(self, model, model_enc_dict, model_ptxt_dict, model_param, model_layer_size_enc, optim, device, \
                         HE_context, pub_context, priv_context, HE_method, args, \
                         n_round, s_mask, mask_indices, top_params):
        
        # Clients get encrypted model at the first round in this version
        if 0 == n_round:
            # s_mask = he_utils.decrypt
            model.load_state_dict(model_ptxt_dict) 
            HE_dec_time = -1
            non_HE_reshape_ctxt_time = -1
            non_HE_recover_model_time = -1 # combine decrypted model + ptxt together
            non_HE_reshape_ptxt_time = -1
            
        elif n_round > 0: # later rounds, the model to be trained is combined by encrypted and plaintext model from the previous round
                        
            # Because layers in model_ptxt_dict are 1D, needs to convert to original shape, and fill 0s to position of encrypted params
            recovered_model_ptxt_dict, non_HE_reshape_ptxt_time = smap_utils.recover_ptxt_shape3(model, model_ptxt_dict, s_mask)

            # For mobilenetv3, mean, var in batch norm are not trainable, but still need to record for load_dict
            for key in model_ptxt_dict:
                if key not in recovered_model_ptxt_dict:
                    # print(f"not in : {key}")
                    recovered_model_ptxt_dict[key] = model_ptxt_dict[key]

            if(args.s_ratio > 0.0):
                model_dec_ctxt_dict, HE_dec_time, non_HE_reshape_ctxt_time = \
                    he_utils.partial_decrypt_model_separate(HE_method, model_enc_dict, model_layer_size_enc, \
                                                            HE_context, priv_context, model.state_dict(), mask_indices, args)# TODO: DEL, only validate if encryption is correct
                
                
                
                # Non-HE time, add models together
                non_HE_recover_model_t_start = time.time()

                if len(recovered_model_ptxt_dict.keys()) > len(model_dec_ctxt_dict):
                    keys = recovered_model_ptxt_dict.keys()
                else:
                    keys = model_dec_ctxt_dict.keys()

                for key in keys: # each layer, save the content from bigger dict to smaller dict
                    if key in model_dec_ctxt_dict:
                        # before = model_dec_ctxt_dict[key]
                        # model_dec[key] = torch.add(model_dec[key], model_ptxt_avg[key]) 
                        model_dec_ctxt_dict[key] = torch.add(model_dec_ctxt_dict[key], recovered_model_ptxt_dict[key]) 
                        # after = model_dec_ctxt_dict[key]
                        # stop =1 # attention: check if plaintext and decrypted models are correctly elementwise-added
                    else:
                        model_dec_ctxt_dict[key] = recovered_model_ptxt_dict[key]
                non_HE_recover_model_t_end = time.time()
                non_HE_recover_model_time = non_HE_recover_model_t_end - non_HE_recover_model_t_start

                model.load_state_dict(model_dec_ctxt_dict) # get decrypted weight

            elif(args.s_ratio == 0.0):
                model_dec_ctxt_dict = {}
                HE_dec_time = -1
                non_HE_reshape_ctxt_time = -1
                non_HE_recover_model_time = -1
                

                model.load_state_dict(recovered_model_ptxt_dict) # get decrypted weight

            if(args.s_ratio == 1.0):
                model_dec_ctxt_dict, HE_dec_time, non_HE_reshape_ctxt_time = \
                    he_utils.partial_decrypt_model_separate(HE_method, model_enc_dict, model_layer_size_enc, \
                                                            HE_context, priv_context, model.state_dict(), mask_indices, args)# TODO: DEL, only validate if encryption is correct

                model.load_state_dict(model_dec_ctxt_dict) # get decrypted weight
                non_HE_recover_model_time = -1
                non_HE_reshape_ptxt_time = -1
            
            # model.load_state_dict(model_dec_ctxt_dict)
            stop = 1

        # originally, the optimizer needs optim.param_groups[0]['params'] = list(client_model.parameters()) but this version uses dict; therefore, assign optim here
        optim.param_groups[0]['params'] = list(model.parameters()) # TODO: 20240828, check if accuracy is increased

        # Train local model
        # Drop client if train set is empty
        if self.loaders['train'] is None:
            if not self.args.quiet: print(f'            No data!')
            return None, 0, 0, None

        # Determine if client is a straggler and drop it if required
        straggler = np.random.binomial(1, self.args.hetero)
        if straggler and self.args.drop_stragglers:
            if not self.args.quiet: print(f'            Dropped straggler!')
            return None, 0, 0, None
        epochs = np.random.randint(1, self.args.epochs) if straggler else self.args.epochs

        # Create training loader
        if self.args.vc_size is not None:
            # Virtual Client (FedVC)
            if len(self.loaders['train'].dataset) >= self.args.vc_size:
                train_idxs_vc = torch.randperm(len(self.loaders['train'].dataset))[:self.args.vc_size]
            else:
                train_idxs_vc = torch.randint(len(self.loaders['train'].dataset), (self.args.vc_size,))
            train_loader = DataLoader(Subset(self.loaders['train'].dataset, train_idxs_vc), batch_size=self.train_bs, shuffle=True)
        else:
            # No Virtual Client
            train_loader = self.loaders['train']

        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)

        # Train new model
        local_train_t_start = time.time()
        model.to(device)
        self.criterion.to(device)
        model.train()
        model_server = deepcopy(model)
        iter = 0
        for epoch in range(epochs):
            loss_sum, loss_num_images, num_images = 0., 0, 0
            for batch, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(examples)
                loss = self.criterion(log_probs, labels)

                if self.args.mu > 0 and epoch > 0:
                    # Add proximal term to loss (FedProx)
                    w_diff = torch.tensor(0., device=device)
                    for w, w_t in zip(model.parameters(), model_server.parameters()):
                        w_diff += torch.pow(torch.norm(w.data - w_t.data), 2)
                        #w.grad.data += self.args.mu * (w.data - w_t.data)
                        w.grad.data += self.args.mu * (w_t.data - w.data)
                    loss += self.args.mu / 2. * w_diff

                loss_sum += loss.item() * len(labels)
                loss_num_images += len(labels)
                num_images += len(labels)

                loss.backward()
                optim.step()

                # After client_stats_every batches...
                if (batch + 1) % client_stats_every == 0:
                    # ...Compute average loss
                    loss_running = loss_sum / loss_num_images

                    # ...Print stats
                    if not self.args.quiet:
                        print('            ' + f'Epoch: {epoch+1}/{epochs}, '\
                                               f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)}), '\
                                               f'Loss: {loss.item():.6f}, ' \
                                               f'Running loss: {loss_running:.6f}')

                    loss_sum, loss_num_images = 0., 0

                iter += 1
        local_train_t_end = time.time()
        local_train_time = local_train_t_end - local_train_t_start
        # local training ends

        # Compute model update, NOT used for HE
        model_update_dict = {}
        for key in model.state_dict():
            model_update_dict[key] = torch.sub(model_server.state_dict()[key], model.state_dict()[key])

        # Perform partial encryption based on sensitivity map
        # unmasked_model_dict: part_ptxt_model_dict: model parameters to be stayed as plaintext
        # masked_model_dict: model parameters to be encrypted
        non_HE_mask_model_t_start = time.time()
        masked_model_dict, part_ptxt_model_dict, unmasked_part_model_dict = smap_utils.mask_model(model, s_mask) # non-HE time, masked models
        # masked_model_dict contains 0 which indicates unsensitive parameters, they are stored in unmasked_part_model_dict
        # For encryption, 0 parameters will be removed from masked_model_dict
        
        # flatten umasked model (ptxt), remove 0s
        
        non_HE_mask_model_t_end = time.time()
        non_HE_mask_model_time = non_HE_mask_model_t_end - non_HE_mask_model_t_start

        # Perform encryption: for each layer, indices of non-zero values are from top_params
        ## Create a dictionary storing mask indices per layer
        non_HE_select_flat_t_start = time.time()
        mask_indices = {}
        mask_multi_indices = {}
        layer_to_enc_params = {} # sensitive parameters to be encrypted from masked_model_dict

        ## create empty lists per layer
        for key in s_mask:
            mask_indices[key] = []
            mask_multi_indices[key] = []
            layer_to_enc_params[key] = []

        for name, _, multi_idx, idx in top_params: # idx in top_params are not in correct order, need multi_idx to get the correct and original position
            # print(f"processing layer {key}...")
            mask_multi_indices[name].append(multi_idx)
            mask_indices[name].append(idx)

        ## collect non-zero weight values of masked model
        if args.s_ratio > 0:
            for layer_name in masked_model_dict:
                non_zero_indices = mask_indices[layer_name] 
                # new, sort non_zero_indices
                non_zero_indices.sort() # sort indices in acending order, not to mess up with order of parameters
                layer_weight_flatten = masked_model_dict[layer_name].cpu().numpy().flatten()
                layer_to_enc_params[layer_name] = [layer_weight_flatten[idx] for idx in non_zero_indices]

        elif args.s_ratio == 0:
            for layer_name in masked_model_dict:
                mask_indices[layer_name] = [0]
                non_zero_indices = mask_indices[layer_name] # always assign 0 as index
                layer_weight_flatten = masked_model_dict[layer_name].cpu().numpy().flatten()
                layer_to_enc_params[layer_name] = [layer_weight_flatten[idx] for idx in non_zero_indices]

        # TODO: pay at to s_ratio == 1.0

        non_HE_select_flat_t_end = time.time()
        non_HE_select_flat_time = non_HE_select_flat_t_end - non_HE_select_flat_t_start

        # for layer_name in layer_to_enc_params:
        #     print(f"len({layer_name}) = {len(layer_to_enc_params[layer_name])}")

        ## encrypt non-zero weight values; model_layer_size_enc: stores num params being encrypted
        if(args.s_ratio > 0.0):
            part_enc_model_dict, part_enc_layer_size, HE_enc_time, non_HE_flat_time = \
                he_utils.partial_encrypt_model_new(HE_method, layer_to_enc_params, HE_context, pub_context, priv_context)
        else:
            part_enc_model_dict = None
            part_enc_layer_size = {}
            HE_enc_time = -1
            non_HE_flat_time = -1

        if args.s_ratio == 1.0: # pure plaintext
            unmasked_part_model_dict = None
        
        if args.s_ratio < 1.0: # partial encryption
            for key in model.state_dict():
                if not key in unmasked_part_model_dict: # copy entire weight for a layer if that is not recorded in the mask
                    unmasked_part_model_dict[key] = deepcopy(model.state_dict()[key]) # original implementation, preserve unencrypted parameters


        return model_update_dict, model.state_dict(), \
            part_enc_model_dict, unmasked_part_model_dict, part_enc_layer_size, mask_indices, \
            len(train_loader.dataset), iter, loss_running, \
            local_train_time, HE_enc_time, non_HE_flat_time, HE_dec_time, non_HE_reshape_ctxt_time, \
            non_HE_recover_model_time, non_HE_reshape_ptxt_time, non_HE_mask_model_time, non_HE_select_flat_time # TODO: combine HE_select_flat_time and flat_time

    
    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
