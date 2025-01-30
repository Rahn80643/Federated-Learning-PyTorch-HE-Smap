import numpy as np
import io, re
import time
from datetime import timedelta
from copy import deepcopy
from contextlib import redirect_stdout

import torch
from torch.nn import CrossEntropyLoss
from torchinfo import summary

import optimizers, schedulers

# for HE
import tenseal as ts
import openfhe
from openfhe import *
from Pyfhel import Pyfhel

# separate encryption and flatten for encrypting models
def separate_layers(HE_method, model, max_slot_size):
    layer_info = {}
    for key in model.state_dict():
        if HE_method == "OpenFHE_CKKS":
            layer_weight = model.state_dict()[key].cpu().numpy().flatten()

        elif HE_method == "TenSeal_CKKS":
            layer_weight = model.state_dict()[key].cpu().numpy().flatten()
    
        elif "Pyfhel_CKKS" == HE_method:
            layer_weight = model.state_dict()[key].cpu().numpy().flatten().astype(np.float64)

        
        if len(layer_weight) > max_slot_size:
            total_splits = int(np.floor(len(layer_weight) / max_slot_size))
            last_split_num = int(np.remainder(len(layer_weight), max_slot_size))
            
            split_count = 0
            for split_idx in range(total_splits):
                split_param_end = split_count + max_slot_size
                splited_layer_params = layer_weight[split_count: split_param_end]
                key_name = f"{key}_split_enc_{split_idx}"
                layer_info[key_name] = splited_layer_params
                split_count = split_param_end

            if last_split_num > 0:
                splited_layer_params = layer_weight[split_count: split_count + last_split_num]
                key_name = f"{key}_split_enc_{total_splits}"
                layer_info[key_name] = splited_layer_params
        else:
            layer_info[key] = layer_weight
    
    return layer_info

def encrypt_layers(HE_method, layer_info, HE_context, pub_context):
    model_enc = {}
    model_layer_size_enc = {}

    for key, layer_weight in layer_info.items():
        if "OpenFHE_CKKS" == HE_method:
            layer_plaintext = HE_context.MakeCKKSPackedPlaintext(layer_weight)
            encrypted_layer = HE_context.Encrypt(pub_context, layer_plaintext)

        elif HE_method == "TenSeal_CKKS":
            encrypted_layer = ts.ckks_vector(HE_context, layer_weight)

        elif HE_method == "Pyfhel_CKKS":
            layer_plaintext = HE_context.encodeFrac(layer_weight)
            encrypted_layer = HE_context.encryptPtxt(layer_plaintext)
        model_enc[key] = encrypted_layer
        model_layer_size_enc[key] = len(layer_weight)

    return model_enc, model_layer_size_enc

def encrypt_model_new(HE_method, model, HE_context, pub_context, priv_context):

    # 2024.07.11, test TenSeal without tearing out all layers
    if "TenSeal_CKKS_without_flatten" == HE_method:
        model_enc = {}
        model_layer_size_enc = {}
        
        for key in model.state_dict():
            flat_t_start = time.time()
            layer_weight = model.state_dict()[key].cpu().numpy().flatten()
            flat_t_end = time.time()
            flat_time = flat_t_end - flat_t_start

            enc_t_start = time.time()
            encrypted_layer = ts.ckks_vector(HE_context, layer_weight)
            model_enc[key] = encrypted_layer
            model_layer_size_enc[key] = len(layer_weight)
            enc_t_end = time.time()
            enc_time = enc_t_end - enc_t_start

    else:
        if "OpenFHE_CKKS" == HE_method:
            max_slot_size = (int)(HE_context.GetRingDimension()/2)

        elif HE_method == "TenSeal_CKKS":
            max_slot_size = 8192//2

        elif HE_method == "Pyfhel_CKKS":
            max_slot_size = (int)(pub_context.get_nSlots())
        
        flat_t_start = time.time()
        layer_info = separate_layers(HE_method, model, max_slot_size) # flatten model
        flat_t_end = time.time()
        flat_time = flat_t_end - flat_t_start
        
        enc_t_start = time.time()
        model_enc, model_layer_size_enc = encrypt_layers(HE_method, layer_info, HE_context, pub_context)
        enc_t_end = time.time()
        enc_time = enc_t_end - enc_t_start


    return  model_enc, model_layer_size_enc, enc_time, flat_time


# HE related functions
# for appending long layer params in aggregated updates
def is_substring_in_keys(data_dict, substring):
    for key in data_dict.keys():
        if substring in key:
            return True
    return False

# separate decrypt and reshape
def group_split_layers(model_enc):
    grouped_dict = {}
    for key, value in model_enc.items():
        prefix = key.split("_split_enc_")[0]
        if prefix in grouped_dict:
            grouped_dict[prefix][key] = value
        else:
            grouped_dict[prefix] = {key: value}
    return grouped_dict


def decrypt_layers(HE_method, model_enc, model_layer_size_enc, HE_context, skContext):
    if HE_method == "TenSeal_CKKS_without_flatten":
        decrypted_layers = {}
        for key, value in model_enc.items():
            layer_size = model_layer_size_enc[key]
            # decrypted_layers[key] = decrypt_layer(HE_context, value, skContext, layer_size)

            decrypted_wrong_size = value.decrypt()
            decrypted = decrypted_wrong_size[:layer_size]
            decrypted_layers[key] = list(map(lambda x: x.real, decrypted))

    else:
        if HE_method == "OpenFHE_CKKS":
            decrypted_layers = {}
            is_splitted = is_substring_in_keys(model_enc, "_split_enc_")

            if is_splitted:
                grouped_dict = group_split_layers(model_enc)
                for group, items in grouped_dict.items():
                    decrypted_group = []
                    for key, value in items.items(): # value: encrypted_layer
                        layer_size = model_layer_size_enc[key]
                        # decrypted_layer = decrypt_layer(HE_context, value, skContext, layer_size)
                        decrypted_wrong_size = HE_context.Decrypt(value, skContext).GetCKKSPackedValue()
                        decrypted = decrypted_wrong_size[:layer_size]
                        decrypted_layer = list(map(lambda x: x.real, decrypted))

                        decrypted_group.extend(decrypted_layer)
                    decrypted_layers[group] = decrypted_group
            else:
                for key, value in model_enc.items():
                    layer_size = model_layer_size_enc[key]
                    # decrypted_layers[key] = decrypt_layer(HE_context, value, skContext, layer_size)

                    decrypted_wrong_size = HE_context.Decrypt(value, skContext).GetCKKSPackedValue()
                    decrypted = decrypted_wrong_size[:layer_size]
                    decrypted_layers[key] = list(map(lambda x: x.real, decrypted))

        elif HE_method == "TenSeal_CKKS":
            decrypted_layers = {}
            is_splitted = is_substring_in_keys(model_enc, "_split_enc_")

            if is_splitted:
                grouped_dict = group_split_layers(model_enc)
                for group, items in grouped_dict.items():
                    decrypted_group = []
                    for key, value in items.items(): # value: encrypted_layer
                        layer_size = model_layer_size_enc[key]
                        # decrypted_layer = decrypt_layer(HE_context, value, skContext, layer_size)
                        decrypted_wrong_size = value.decrypt()
                        decrypted = decrypted_wrong_size[:layer_size]
                        decrypted_layer = list(map(lambda x: x.real, decrypted))

                        decrypted_group.extend(decrypted_layer)
                    decrypted_layers[group] = decrypted_group
            else:
                for key, value in model_enc.items():
                    layer_size = model_layer_size_enc[key]
                    # decrypted_layers[key] = decrypt_layer(HE_context, value, skContext, layer_size)

                    decrypted_wrong_size = value.decrypt()
                    decrypted = decrypted_wrong_size[:layer_size]
                    decrypted_layers[key] = list(map(lambda x: x.real, decrypted))

        elif HE_method == "Pyfhel_CKKS":
            decrypted_layers = {}
            is_splitted = is_substring_in_keys(model_enc, "_split_enc_")

            if is_splitted:
                grouped_dict = group_split_layers(model_enc)
                for group, items in grouped_dict.items():
                    decrypted_group = []
                    for key, value in items.items(): # value: encrypted_layer
                        layer_size = model_layer_size_enc[key]
                        # decrypted_layer = decrypt_layer(HE_context, value, skContext, layer_size)
                        decrypted_wrong_size = skContext.decryptFrac(value).tolist()
                        decrypted = decrypted_wrong_size[:layer_size]
                        decrypted_layer = list(map(lambda x: x.real, decrypted))

                        decrypted_group.extend(decrypted_layer)
                    decrypted_layers[group] = decrypted_group
            else:
                for key, value in model_enc.items():
                    layer_size = model_layer_size_enc[key]
                    # decrypted_layers[key] = decrypt_layer(HE_context, value, skContext, layer_size)

                    decrypted_wrong_size = skContext.decryptFrac(value).tolist()
                    decrypted = decrypted_wrong_size[:layer_size]
                    decrypted_layers[key] = list(map(lambda x: x.real, decrypted))

    return decrypted_layers

def reshape_decrypted_model(decrypted_layers, model_param, args):
    reshaped_model = {}
    for key, decrypted_layer in decrypted_layers.items():
        decrypted_tensor = torch.tensor(decrypted_layer).to(args.device)
        reshaped_model[key] = decrypted_tensor.reshape(model_param[key].shape)
    return reshaped_model

def decrypt_model_separate(HE_method, model_enc, model_layer_size_enc, HE_context, skContext, model_param, args):
    dec_t_start = time.time()
    decrypted_layers = decrypt_layers(HE_method, model_enc, model_layer_size_enc, HE_context, skContext)
    dec_t_end = time.time()
    dec_time = dec_t_end - dec_t_start

    reshape_t_start = time.time()
    model_dec = reshape_decrypted_model(decrypted_layers, model_param, args)
    reshape_t_end = time.time()
    reshape_time = reshape_t_end - reshape_t_start

    return model_dec, dec_time, reshape_time

def partial_encrypt_model_new(HE_method, enc_param_dict, HE_context, pub_context, priv_context):

    if "TenSeal_CKKS_without_flatten" == HE_method:
        model_enc = {}
        model_layer_size_enc = {}
        
        for key in enc_param_dict:
            flat_t_start = time.time()
            layer_weight = enc_param_dict[key]
            flat_t_end = time.time()
            flat_time = flat_t_end - flat_t_start

            enc_t_start = time.time()
            if(len(layer_weight) > 0):
                encrypted_layer = ts.ckks_vector(HE_context, layer_weight)

                model_enc[key] = encrypted_layer
                model_layer_size_enc[key] = len(layer_weight)
            enc_t_end = time.time()
            enc_time = enc_t_end - enc_t_start

    return  model_enc, model_layer_size_enc, enc_time, flat_time

# for partial decryption; potential optimization
def reshape_partial_decrypted_model(decrypted_layers, masked_model, mask_indices, args):
    reshaped_model = {}
    for key, decrypted_layer in decrypted_layers.items():
        # decrypted_tensor = torch.tensor(decrypted_layer).to(args.device)
        # decrypted_tensor = torch.tensor(decrypted_layer)
        # flatten_layer = np.zeros(len(masked_model[key].cpu().numpy().flatten())) # initialize flatten layer with 0s
        flatten_layer2 = torch.zeros(masked_model[key].shape.numel(), device=args.device) # verify

        # for value, index in zip(decrypted_tensor, mask_indices[key]): # iterate through all indices in this layer, inefficient
        #     flatten_layer[index] = value # recover parameters to their original position

        flatten_layer2.scatter_(0, torch.tensor(mask_indices[key], device=args.device), torch.tensor(decrypted_layer, device=args.device))

        # reshaped_model[key] = torch.tensor(flatten_layer).reshape(masked_model[key].shape).to(args.device)
        reshaped_model[key] = flatten_layer2.reshape(masked_model[key].shape)
    return reshaped_model



def partial_decrypt_model_separate(HE_method, model_enc, model_layer_size_enc, HE_context, skContext, masked_model, mask_indices, args):
    # if not args.quiet:
    #     print("partial decryption")
    # masked_model: only provides layer shapes
    dec_t_start = time.time()
    decrypted_layers = decrypt_layers(HE_method, model_enc, model_layer_size_enc, HE_context, skContext)
    dec_t_end = time.time()
    dec_time = dec_t_end - dec_t_start

    reshape_t_start = time.time()
    model_dec = reshape_partial_decrypted_model(decrypted_layers, masked_model, mask_indices, args)
    reshape_t_end = time.time()
    reshape_time = reshape_t_end - reshape_t_start

    return model_dec, dec_time, reshape_time


def average_updates_HE_OpenFHE_CKKS(w_he, HE_context):
    # print("executing average_updates_HE_TenSeal_CKKS....")
    # print(f"len(w_he): {len(w_he)}, len(w_avg.keys()): {len(w_he[0].keys())}")

    # w_avg = deepcopy(w_he[0]) # error: can't pickle openfhe.Ciphertext objects
    w_avg = w_he[0]
    # print("deep copy finished")
    agg_t_start = time.time()
    for key in w_avg.keys():
        for i in range(1, len(w_he)): # for each model
            w_avg[key] = HE_context.EvalAdd(w_avg[key], w_he[i][key])

        w_avg[key] = HE_context.EvalMult(w_avg[key], (1/(len(w_he))))
    
    agg_t_end = time.time()
    agg_time = agg_t_end - agg_t_start
    # print("finished executing average_updates_HE")
    return w_avg, agg_time

def average_updates_HE_TenSeal_CKKS(w_he):
    
    # print("executing average_updates_HE_TenSeal_CKKS....")
    # print(f"len(w_he): {len(w_he)}, len(w_avg.keys()): {len(w_he[0].keys())}")

    # w_avg = deepcopy(w_he[0]) # include this time?
    w_avg = w_he[0] # deep copy consumes lots of memory
    # print("deep copy finished")
    agg_t_start = time.time()
    for key in w_avg.keys(): # WHY NOT in the loop after the 4th round?? wrong type?
        for i in range(1, len(w_he)): # for each model
            w_avg[key] = w_avg[key] + w_he[i][key]

        before = w_avg[key].decrypt()
        w_avg[key] = w_avg[key] * (1/(len(w_he)))
        # after = w_avg[key].decrypt()
        # divided = [bef * (1/(len(w_he))) for bef in before]
        # stop =1
    
    agg_t_end = time.time()
    agg_time = agg_t_end - agg_t_start
    # print("finished executing average_updates_HE")
    return w_avg, agg_time

def average_updates_HE_Pyfhel_CKKS(w_he):
    # print("executing average_updates_HE_TenSeal_CKKS....")
    # print(f"len(w_he): {len(w_he)}, len(w_avg.keys()): {len(w_he[0].keys())}")

    # w_avg = deepcopy(w_he[0]) # For public key encrypted models, they cannot be deep copied because of lacking values of qi
    w_avg = w_he[0]
    # print("deep copy finished")
    agg_t_start = time.time()
    for key in w_avg.keys():
        for i in range(1, len(w_he)): # for each model
            w_avg[key] = w_avg[key]+ w_he[i][key] # homomorphic addition

        w_avg[key] = w_avg[key]* (1/(len(w_he))) # homomorphic multiplication with a constant
    
    agg_t_end = time.time()
    agg_time = agg_t_end - agg_t_start
    # print("finished executing average_updates_HE")
    return w_avg, agg_time