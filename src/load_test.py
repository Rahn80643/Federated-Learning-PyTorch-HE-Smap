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
from time import time
from datetime import timedelta
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import datasets, models, optimizers, schedulers
from options import args_parser
from utils import average_updates, average_updates_mod, exp_details, get_acc_avg, printlog_stats
from datasets_utils import Subset, get_datasets_fig
from sampling import get_splits, get_splits_fig
from client import Client

# modified, for HE
import os
import keygen
import he_utils as he_utils 
import openfhe
from openfhe import *
import tenseal as ts

import pandas as pd
from openpyxl import load_workbook

if __name__ == '__main__':
    # Start timer
    start_time = time()

    # Parse arguments and create/load checkpoint
    args = args_parser()
    datetime =  "2024-07-11_16-19-53"
    mainfolder = "/home/renyi/Research/Federated-Learning-PyTorch_HE_formal/save/20240711"
    lib = "TenSeal_CKKS_without_flatten"
    data_folder = mainfolder + "/" + lib + "_" + datetime + "/"

    # checkpoint = torch.load(f'save/{args.name}') # error: CKKS scheme requires a list of prime sizes (qi_sizes) or primes (qi) to be set if the model is using public key only
    # checkpoint = torch.load(f"{mainfolder}/Pyfhel_CKKS_{datetime}MobileNetv1/{datetime}-final")
    checkpoint = torch.load(f"{data_folder}/{datetime}-final")
    
    rounds = args.rounds
    iters = args.iters
    device =args.device
    args = checkpoint['args']
    args.resume = True
    args.rounds = rounds
    args.iters = iters
    args.device = device
    HE_method = checkpoint["HE_method"]

    ## Initialize RNGs and ensure reproducibility
    if args.seed is not None:
        environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

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
    
    # # model.load_state_dict(checkpoint['model_state_dict'])
    model_layer_size_enc = checkpoint['model_layer_size']
    # # TODO: public and secret key are empty and cannot be decrypted (weird)
    # # is_public_key_empty, is_secret_key_empty
    key_folder = checkpoint["HE_key_folder"]

    if HE_method == "OpenFHE_CKKS":
        # datafolder = checkpoint['HE_key_folder']
        datafolder = os.path.join(mainfolder, f"OpenFHE_CKKS_{datetime}")
        model_enc_dir = os.path.join(datafolder, "model_enc")

        # load saved keys
        HE_context, res1 = DeserializeCryptoContext(datafolder + "/CKKS_context.txt", BINARY)
        HE_pk, res2 = DeserializePublicKey(datafolder + "/pubKey.txt", BINARY)
        HE_sk, res3 = DeserializePrivateKey(datafolder + "/privKey.txt", BINARY)

        loaded_dict = {}
        for key in model_layer_size_enc:
            layer_deserialized, layer_res = DeserializeCiphertext(os.path.join(model_enc_dir, key), BINARY)
            loaded_dict[key] = layer_deserialized
        
        loaded_dec_model, _ = he_utils.decrypt_model(HE_method, loaded_dict, model_layer_size_enc, HE_context, HE_sk, model.state_dict(), args)# TODO: DEL, only validate if encryption is correct
        model.load_state_dict(loaded_dec_model)
        clients = checkpoint['clients']
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
        
    elif HE_method == "TenSeal_CKKS" or HE_method == "TenSeal_CKKS_without_flatten":
        model_enc_dir = os.path.join(data_folder, "model_enc")

        # load saved keys
        with open(os.path.join(data_folder, "ser_context"), "rb") as f:
            read_context = f.read()
            HE_context = ts.context_from(read_context)
        
        with open(os.path.join(data_folder, "ser_pubKey"), "rb") as f:
            read_context = f.read()
            HE_pk = ts.context_from(read_context)
        
        with open(os.path.join(data_folder, "ser_privKey"), "rb") as f:
            read_context = f.read()
            HE_sk = ts.context_from(read_context)

        # load weight
        loaded_dict = {}
        for key in model_layer_size_enc:
            with open(os.path.join(model_enc_dir, key), "rb") as f:
                load_layer = f.read() # byte
                # abc = ts.CKKSVector.load(HE_context, load_layer)
                load_layer_linked = ts.ckks_vector_from(HE_sk, load_layer) # load_layer.link_context(HE_context)
                loaded_dict[key] = load_layer_linked
        
        # loaded_dec_model = he_utils.decrypt_model(HE_method, loaded_dict, model_layer_size_enc, HE_context, HE_sk, model.state_dict(), args)# TODO: DEL, only validate if encryption is correct
        print("decrypting model...")
        loaded_dec_model, dec_time, reshape_time = he_utils.decrypt_model_separate(HE_method, loaded_dict, model_layer_size_enc, HE_context, HE_sk, model.state_dict(), args)
        print("decryption completed")
        model.load_state_dict(loaded_dec_model)
        clients = checkpoint['clients']
        print("calculating test accuracy")
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)

    elif HE_method == "Pyfhel_CKKS":
        model_dict = torch.load(f"{mainfolder}/Pyfhel_CKKS_{datetime}/model_enc-final.model")
        model_HE_load = model_dict # checkpoint['model_enc']
        from Pyfhel import Pyfhel
        HE_sk = Pyfhel()
        with open(os.path.join(key_folder, "context"), 'rb') as f:
            load_context = HE_sk.from_bytes_context(f.read())
        
        with open(os.path.join(key_folder, "privKey"), 'rb') as f:
            load_context = HE_sk.from_bytes_secret_key(f.read())

        with open(os.path.join(key_folder, "pubKey"), 'rb') as f:
            load_pk = HE_sk.from_bytes_public_key(f.read())

        model_dec, _ = he_utils.decrypt_model(HE_method, model_HE_load, model_layer_size_enc, HE_sk, HE_sk, model.state_dict(), args)
        model.load_state_dict(model_dec)
        clients = checkpoint['clients']
        acc_avg = get_acc_avg(acc_types, clients, model, args.device)
    
    print(f"decrypted {checkpoint['args'].model} model trained in {datetime} with {checkpoint['HE_method']}")
    print(f"acc_avg:{acc_avg}")
    stop = 1