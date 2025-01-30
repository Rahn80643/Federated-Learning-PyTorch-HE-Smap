#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
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

def process_dict_client(d, title):
    df = pd.DataFrame(d).T
    df = df.reindex(columns=[0, 1, 2])
    df.index = [1, 2, 3, 4, 5]
    # df.index = range(1, 50+1)
    df.columns = [1, 2, 3]
    return df

def process_dict_server(d, title):
    df = pd.DataFrame(list(d.values()), columns=[title])
    # df.index = [1, 2, 3, 4, 5]
    df.index = list(d.keys())  # Use the original keys as index
    return df

if __name__ == '__main__':

    # 2024.02.13: generate keys from PHE, key Authority Generate Key
    # keypair = paillier.generate_paillier_keypair(n_length=config['key_length'])
    # pubKey, privKey = keypair
    # keypair generation
    # TODO: parameterize in option.py

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, 
                        default="/home/renyi/Research/Federated-Learning-PyTorch_HE_formal_mod/save/TenSeal_CKKS_without_flatten_ConvNet05_8192_52", help="data root folder")
    parser.add_argument("--datetime", type=str, default="2024-10-01_19-22-41")
    # parser.add_argument("--ring_dim", type=int, default=8192)


    args = parser.parse_args()


    datetime = args.datetime
    mainfolder = args.root + "_" + datetime
    

    # checkpoint = torch.load(f'save/{args.name}') # error: CKKS scheme requires a list of prime sizes (qi_sizes) or primes (qi) to be set if the model is using public key only
    # checkpoint = torch.load(f"{mainfolder}/Pyfhel_CKKS_{datetime}MobileNetv1/{datetime}-final")
    checkpoint = torch.load(os.path.join(mainfolder, datetime+"-final"))
    HE_method = checkpoint["HE_method"]

    training_info = {}
    training_info["HE_method"] = checkpoint["HE_method"]
    training_info["dataset"] = checkpoint["args"].dataset
    training_info["model"] = checkpoint["args"].model
    training_info["FL_round"] = checkpoint["args"].rounds
    training_info["num_clients"] = checkpoint["args"].num_clients
    training_info["optim"] = checkpoint["args"].optim
    training_info["train_bs"] = checkpoint["args"].train_bs
    training_info["local_epochs"] = checkpoint["args"].epochs

    output_name = HE_method + "_" + datetime + ".xlsx"
    output_name_avg = HE_method + "_" + datetime + "_avg" + ".xlsx"

    # for exporting to csv file
    # Process each dictionary
    # keygen_df = checkpoint['keygen_time']
    # training_info_df = pd.DataFrame(list(training_info.items()), columns=['Parameter', 'Value'])
    # keygen_df = pd.DataFrame({'Keygen time (s)': [checkpoint['keygen_time']]})
    # flat_df = process_dict_client(checkpoint['flat_time_dicts'], "Flat time (s)")
    # flat_avg_df = pd.DataFrame({np.average(flat_df.values)})
    # enc_df = process_dict_client(checkpoint['enc_time_dicts'], "Encryption time (s)")
    # enc_avg_df = pd.DataFrame({np.average(enc_df.values)})
    # dec_df = process_dict_client(checkpoint['dec_time_dicts'], "Decryption time (s)")
    # dec_avg_df = pd.DataFrame({np.average(dec_df.values)})
    # reshape_df = process_dict_client(checkpoint['reshape_time_dicts'], "Flat time (s)")
    # reshape_avg_df = pd.DataFrame({np.average(reshape_df.values)})
    # train_df = process_dict_client(checkpoint['local_train_time_dicts'], "Local training time (s)")
    # train_avg_df = pd.DataFrame({np.average(train_df.values)})
    # agg_df = process_dict_server(checkpoint['server_agg_times'], "Server aggregation time (s)")
    # agg_avg_df = pd.DataFrame({np.average(agg_df.values)})
    # non_HE_df = process_dict_server(checkpoint['non_HE_times'], "Server non HE time (s)")
    # non_HE_avg_df = pd.DataFrame({np.average(non_HE_df.values)})

    stop = 1
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    # with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
    #     training_info_df.to_excel(writer, sheet_name='Training Info', index=False)
    #     keygen_df.to_excel(writer, sheet_name='Ser keygen time', startrow=1, index=False)
    #     flat_avg_df.to_excel(writer, sheet_name='Flatten avg time', startrow=1)
    #     enc_avg_df.to_excel(writer, sheet_name='Encryption avg time', startrow=1)
    #     dec_avg_df.to_excel(writer, sheet_name='Decryption avg time', startrow=1)
    #     reshape_avg_df.to_excel(writer, sheet_name='Reshape avg time', startrow=1)
    #     train_avg_df.to_excel(writer, sheet_name='Local training avg time', startrow=1)
    #     agg_avg_df.to_excel(writer, sheet_name='SAgg avg time', startrow=1)
    #     non_HE_avg_df.to_excel(writer, sheet_name='SNonHE avg Time', startrow=1)

    #     flat_df.to_excel(writer, sheet_name='Flatten time', startrow=1)
    #     enc_df.to_excel(writer, sheet_name='Encryption time', startrow=1)
    #     dec_df.to_excel(writer, sheet_name='Decryption time', startrow=1)
    #     reshape_df.to_excel(writer, sheet_name='Reshape time', startrow=1)
    #     train_df.to_excel(writer, sheet_name='Local training time', startrow=1)
    #     agg_df.to_excel(writer, sheet_name='SAggTime', startrow=1)
    #     non_HE_df.to_excel(writer, sheet_name='SNonHETime', startrow=1)

    keygen_df = checkpoint['keygen_time']
    training_info_df = pd.DataFrame(list(training_info.items()), columns=['Parameter', 'Value'])
    keygen_df = pd.DataFrame({'Keygen time (s)': [checkpoint['keygen_time']]})

    nonHE_flat_df = process_dict_client(checkpoint['non_HE_flat_time_dicts'], "NFlat time (s)")
    filtered_nonHE_flat_df = nonHE_flat_df[nonHE_flat_df >= 0]
    nonHE_flat_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_flat_df.values)})

    nonHE_select_flat_df = process_dict_client(checkpoint['non_HE_select_flat_time_dicts'], "NSelect flat time (s)")
    filtered_nonHE_select_flat_df = nonHE_select_flat_df[nonHE_select_flat_df >= 0]
    nonHE_select_flat_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_select_flat_df.values)})

    # discard, save information, and nothing to do with FL training
    nonHE_df = process_dict_server(checkpoint['non_HE_times'], "NServer non HE time (s)")
    filtered_non_HE_df = nonHE_df[nonHE_df >= 0]
    nonHE_avg_df = pd.DataFrame({np.nanmean(filtered_non_HE_df.values)})

    nonHE_recover_model_df = process_dict_client(checkpoint['non_HE_recover_model_time_dicts'], "NRecover model time (s)")
    filtered_nonHE_recover_model_df = nonHE_recover_model_df[nonHE_recover_model_df >= 0]
    nonHE_recover_model_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_recover_model_df.values)})

    nonHE_mask_model_df = process_dict_client(checkpoint['non_HE_mask_model_time_dicts'], "NMask model time (s)")
    filtered_nonHE_mask_model_df = nonHE_mask_model_df[nonHE_mask_model_df >= 0]
    nonHE_mask_model_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_mask_model_df.values)})

    nonHE_reshape_ptxt_df = process_dict_client(checkpoint['non_HE_reshape_ptxt_time_dicts'], "NReshape ptxt time (s)")
    filtered_nonHE_reshape_ptxt_df = nonHE_reshape_ptxt_df[nonHE_reshape_ptxt_df >= 0]
    nonHE_reshape_ptxt_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_reshape_ptxt_df.values)})

    nonHE_reshape_ctxt_df = process_dict_client(checkpoint['non_HE_reshape_ctxt_time_dicts'], "NReshape ctxt time (s)")
    filtered_nonHE_reshape_ctxt_df = nonHE_reshape_ctxt_df[nonHE_reshape_ctxt_df >= 0]
    nonHE_reshape_ctxt_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_reshape_ctxt_df.values)})

    nonHE_agg_ptxt_df = process_dict_server(checkpoint['server_nonHE_ptxt_agg_times'], "NServer aggregation ptxt time (s)")
    filtered_nonHE_agg_ptxt_df = nonHE_agg_ptxt_df[nonHE_agg_ptxt_df >= 0]
    nonHE_agg_ptxt_avg_df = pd.DataFrame({np.nanmean(filtered_nonHE_agg_ptxt_df.values)})


    HE_enc_df = process_dict_client(checkpoint['HE_enc_time_dicts'], "HEncryption time (s)")
    filtered_HE_enc_df = HE_enc_df[HE_enc_df >= 0]
    HE_enc_avg_df = pd.DataFrame({np.nanmean(filtered_HE_enc_df.values)})
    
    HE_dec_df = process_dict_client(checkpoint['HE_dec_time_dicts'], "HDecryption time (s)")
    filtered_HE_dec_df = HE_dec_df[HE_dec_df >= 0]
    HE_dec_avg_df = pd.DataFrame({np.nanmean(filtered_HE_dec_df.values)})
    
    HE_agg_ctxt_df = process_dict_server(checkpoint['server_HE_ctxt_agg_times'], "HServer aggregation ctxt time (s)")
    filtered_HE_agg_ctxt_df = HE_agg_ctxt_df[HE_agg_ctxt_df >= 0]
    HE_agg_ctxt_avg_df = pd.DataFrame({np.nanmean(filtered_HE_agg_ctxt_df.values)})

    train_df = process_dict_client(checkpoint['local_train_time_dicts'], "Local training time (s)")
    train_avg_df = pd.DataFrame({np.nanmean(train_df.values)})
    

    stop = 1
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(output_name, engine='openpyxl') as writer:
        training_info_df.to_excel(writer, sheet_name='Training Info', index=False)
        keygen_df.to_excel(writer, sheet_name='HSer keygen', startrow=1, index=False)
        HE_enc_df.to_excel(writer, sheet_name='Henc', startrow=1)
        HE_dec_df.to_excel(writer, sheet_name='Hdec', startrow=1)
        HE_agg_ctxt_df.to_excel(writer, sheet_name='Hagg ctxt', startrow=1)
        nonHE_flat_df.to_excel(writer, sheet_name='NFlatten', startrow=1)
        nonHE_select_flat_df.to_excel(writer, sheet_name='NFlatten select', startrow=1)
        nonHE_df.to_excel(writer, sheet_name='NSNonHE', startrow=1)
        nonHE_recover_model_df.to_excel(writer, sheet_name='Nrecover', startrow=1)
        nonHE_mask_model_df.to_excel(writer, sheet_name='Nmask model', startrow=1)
        nonHE_reshape_ptxt_df.to_excel(writer, sheet_name='N eshapt ptxt', startrow=1)
        nonHE_reshape_ctxt_df.to_excel(writer, sheet_name='Nreshapt ctxt', startrow=1)
        nonHE_agg_ptxt_df.to_excel(writer, sheet_name='Nagg ptxt', startrow=1)

        train_df.to_excel(writer, sheet_name='Local training', startrow=1)


    nonHE_total = nonHE_flat_avg_df.values + nonHE_select_flat_avg_df.values + nonHE_recover_model_avg_df.values + \
        nonHE_mask_model_avg_df.values + nonHE_reshape_ptxt_avg_df.values + nonHE_reshape_ctxt_avg_df.values + nonHE_agg_ptxt_avg_df.values

    HE_total = keygen_df.values + HE_enc_avg_df.values + HE_dec_avg_df.values + HE_agg_ctxt_avg_df.values

    nonHE_total = pd.DataFrame(nonHE_total[0]) # remove np.array([[value]])
    HE_total = pd.DataFrame(HE_total[0])

    with pd.ExcelWriter(output_name_avg, engine='openpyxl') as writer:
        training_info_df.to_excel(writer, sheet_name='Training Info', index=False)
        nonHE_total.to_excel(writer, sheet_name='nonHE total', index=False)
        HE_total.to_excel(writer, sheet_name='HE total', index=False)

        keygen_df.to_excel(writer, sheet_name='HSer keygen', startrow=1, index=False)
        HE_enc_avg_df.to_excel(writer, sheet_name='Henc avg', startrow=1)
        HE_dec_avg_df.to_excel(writer, sheet_name='Hdec avg', startrow=1)
        HE_agg_ctxt_avg_df.to_excel(writer, sheet_name='Hagg ctxt avg', startrow=1)
        
        nonHE_flat_avg_df.to_excel(writer, sheet_name='NFlatten avg', startrow=1)
        nonHE_select_flat_avg_df.to_excel(writer, sheet_name='NFlatten select avg', startrow=1)
        # nonHE_avg_df.to_excel(writer, sheet_name='NSNonHE avg Not use', startrow=1)
        nonHE_recover_model_avg_df.to_excel(writer, sheet_name='Nrecover avg', startrow=1)
        nonHE_mask_model_avg_df.to_excel(writer, sheet_name='Nmask model avg', startrow=1)
        nonHE_reshape_ptxt_avg_df.to_excel(writer, sheet_name='N eshapt ptxt avg', startrow=1)
        nonHE_reshape_ctxt_avg_df.to_excel(writer, sheet_name='Nreshapt ctxt avg', startrow=1)
        nonHE_agg_ptxt_avg_df.to_excel(writer, sheet_name='Nagg ptxt avg', startrow=1)

        

        train_avg_df.to_excel(writer, sheet_name='Local training avg', startrow=1)
    # 
    print(f"Data exported to {output_name}")