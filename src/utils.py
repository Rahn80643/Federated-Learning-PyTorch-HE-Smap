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

import io, re
from time import time
from datetime import timedelta
from copy import deepcopy
from contextlib import redirect_stdout

import torch
from torch.nn import CrossEntropyLoss
from torchinfo import summary

import optimizers, schedulers


types_pretty = {'train': 'training', 'valid': 'validation', 'test': 'test'}

class Scheduler():
    def __str__(self):
        sched_str = '%s (\n' % self.name
        for key in vars(self).keys():
            if key != 'name':
                value = vars(self)[key]
                if key == 'optimizer': value = str(value).replace('\n', '\n        ').replace('    )', ')')
                sched_str +=  '    %s: %s\n' % (key, value)
        sched_str += ')'
        return sched_str

def average_s_map(smap_list):
    print("average_s_map")
    smap_avg = deepcopy(smap_list[0])
    avg_state_dict = smap_avg.state_dict()
    
    for key in avg_state_dict.keys():  # each layer
        for i in range(1, len(smap_list)):  # for each model
            print(f" {key}, map idx: {i}")
            abc = avg_state_dict[key] + smap_list[i].state_dict()[key]
            avg_state_dict[key] += smap_list[i].state_dict()[key]
            stop =1

    for key in avg_state_dict.keys():  # each layer  
        if avg_state_dict[key].dtype != torch.int64:
            avg_state_dict[key] /= len(smap_list) # result type Float can't be cast to the desired output type Long
        else: 
            avg_state_dict[key] = torch.round(avg_state_dict[key]/len(smap_list)).to(torch.int64) # for 'model.features.0.1.num_batches_tracked'
    
    smap_avg.load_state_dict(avg_state_dict)
    return smap_avg



# Original FedAvg, considering #data points
def average_updates(w, n_k):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key], alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg

# sum up local updates and take average, number of data points is not used
def average_updates_mod(w):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys(): # each layer
        for i in range(1, len(w)): # for each model
            w_avg[key] = torch.add(w_avg[key], w[i][key]) 
        
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

def inference(model, loader, device):
    if loader is None:
        return None, None

    criterion = CrossEntropyLoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs = model(examples)
            loss += criterion(log_probs, labels).item() * len(labels)
            _, pred_labels = torch.max(log_probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    loss /= total

    return accuracy, loss

def get_acc_avg(acc_types, clients, model, device):
    acc_avg = {}
    for type in acc_types:
        acc_avg[type] = 0.
        num_examples = 0
        for client_id in range(len(clients)):
            acc_client, _ = clients[client_id].inference(model, type=type, device=device)
            if acc_client is not None:
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                num_examples += len(clients[client_id].loaders[type].dataset)
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None

    return acc_avg

def printlog_stats(quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, iters):
    if not quiet:
        print(f'        Iteration: {iter}', end='')
        if iters is not None: print(f'/{iters}', end='')
        print()
        print(f'        Learning rate: {lr}')
        print(f'        Average running loss: {loss_avg:.6f}')
        for type in acc_types:
            print(f'        Average {types_pretty[type]} accuracy: {acc_avg[type]:.3%}')

    if logger is not None:
        logger.add_scalar('Learning rate (Round)', lr, round)
        logger.add_scalar('Learning rate (Iteration)', lr, iter)
        logger.add_scalar('Average running loss (Round)', loss_avg, round)
        logger.add_scalar('Average running loss (Iteration)', loss_avg, iter)
        for type in acc_types:
            logger.add_scalars('Average accuracy (Round)', {types_pretty[type].capitalize(): acc_avg[type]}, round)
            logger.add_scalars('Average accuracy (Iteration)', {types_pretty[type].capitalize(): acc_avg[type]}, iter)
        logger.flush()

def exp_details(args, model, datasets, splits):
    if args.device == 'cpu':
        device = 'CPU'
    else:
        device = str(torch.cuda.get_device_properties(args.device))
        device = (', ' + re.sub('_CudaDeviceProperties\(|\)', '', device)).replace(', ', '\n            ')

    input_size = (args.train_bs,) + tuple(datasets['train'][0][0].shape)
    # if args.model == "resnet18":
    #     input_size = (args.train_bs,) + (3, 224, 224) # refer to self.resize = Resize(224) for resnet18, resnet34, and resnet50
    summ = str(summary(model, input_size, depth=10, verbose=0, col_names=['output_size','kernel_size','num_params','mult_adds'], device=args.device))
    summ = '        ' + summ.replace('\n', '\n        ')

    optimizer = getattr(optimizers, args.optim)(model.parameters(), args.optim_args)
    scheduler = getattr(schedulers, args.sched)(optimizer, args.sched_args)

    if args.centralized:
        algo = 'Centralized'
    else:
        if args.fedsgd:
            algo = 'FedSGD'
        else:
            algo = 'FedAvg'
        if args.server_momentum:
            algo += 'M'
        if args.fedir:
            algo += ' + FedIR'
        if args.vc_size is not None:
            algo += ' + FedVC'
        if args.mu:
            algo += ' + FedProx'
        if args.drop_stragglers:
            algo += ' (Drop Stragglers)'

    f = io.StringIO()
    with redirect_stdout(f):
        print('Experiment summary:')
        print(f'    Algorithm:')
        print(f'        Algorithm: {algo}')
        print(f'        ' + (f'Rounds: {args.rounds}' if args.iters is None else f'Iterations: {args.iters}'))
        print(f'        Clients: {args.num_clients}')
        print(f'        Fraction of clients: {args.frac_clients}')
        print(f'        Client epochs: {args.epochs}')
        print(f'        Training batch size: {args.train_bs}')
        print(f'        System heterogeneity: {args.hetero}')
        print(f'        Server learning rate: {args.server_lr}')
        print(f'        Server momentum (FedAvgM): {args.server_momentum}')
        print(f'        Virtual client size (FedVC): {args.vc_size}')
        print(f'        Mu (FedProx): {args.mu}')
        print()

        print('    Dataset and split:')
        print('        Training set:')
        print('            ' + str(datasets['train']).replace('\n','\n            '))
        if datasets['valid'] is not None:
            print('        Validation set:')
            print('            ' + str(datasets['valid']).replace('\n','\n            '))
        print('        Test set:')
        print('            ' + str(datasets['test']).replace('\n','\n            '))
        print(f'        Identicalness: {args.iid} (EMD = {splits["train"].emd["class"]})')
        print(f'        Balance: {args.balance} (EMD = {splits["train"].emd["client"]})')
        print()

        print('    Scheduler: %s' % (str(scheduler).replace('\n', '\n    ')))
        print()

        print('    Model:')
        print(summ)
        print()

        print('    Other:')
        print(f'        Test batch size: {args.test_bs}')
        print(f'        Random seed: {args.seed}')
        print(f'        Device: {device}')

    return f.getvalue()

def measure_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_bytes = param_size + buffer_size
    return size_all_bytes