import time
import os
import copy
import numpy as np
import torch
from functorch import vmap
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import torch.utils.checkpoint as checkpoint



# input: training data loader, model
# output: calculated averaged smap based on magnitude
def calculate_smap_magnitude(train_loader, batchsize, model, criterion, device):
    # s_maps_magitude = [{} for _ in range(len(train_loader.dataset))] # DEL: only for validate if smaps are sum up and taken average correctly
    s_maps_mag_sum = copy.deepcopy(model.state_dict()) # Accumulate absolute magnitude-based gradient values

    smap_batch_times = []

    # make sure all parameters for s_map_sum are 0 instead of randomly initialized
    for key in s_maps_mag_sum:
        nn.init.zeros_(s_maps_mag_sum[key])

    # Verify that all parameters are initialized to 0
    for key in s_maps_mag_sum:
        print(f"{key}; {s_maps_mag_sum[key].shape}")

    num_images = 0

    # Check correctness of loop, iterate all batches
    smap_jm_start_t = time.time()
    for batch, (x, y) in enumerate(train_loader):
        jm_batch_start_t = time.time()
        # if batch < len(train_loader): 
        if batch < 2: # for testing desired #batches
            print(f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)})')

            x, y = x.to(device), y.to(device)
            num_images += len(y)
            output = model(x)
            loss = criterion(output, y.long()) # contains all loss values for all data points, because the loss function has "reduction='none'"

            # Calculate magnitude-based gradient of the model, finish it layer by layer 
            param_layer_grads_batch = [{} for _ in range(train_loader.batch_size)] # for storing models of gradients in a batch
            # for (param_layer, key) in zip(model.parameters(), model.state_dict()): # model.parameters(), model.state_dict() not in the same order!
            for key, param_layer in model.named_parameters():
                # Calculate gradient of loss w.r.t layer parameters, the result is [#batchsize, #layer shape] after loop
                for loss_idx in range(len(loss)):
                    indiv_loss = loss[loss_idx]
                    param_layer_grad = torch.autograd.grad(outputs=indiv_loss, inputs=param_layer, create_graph=True)[0]
                    param_layer_grads_batch[loss_idx][key] = param_layer_grad.detach().clone()
            # Finish calculating the model for ONE batch

            # param_layer_grads_batch now has a batch of gradients, iterate through all gradients, calculate their absolute values and normalize
            for batIdx in range(len(param_layer_grads_batch)):
                model_grad = param_layer_grads_batch[batIdx] # get a model grad in one datum

                # iterate through all layers to get sum of absolute gradient as sensitivity total
                sensitivity_abs_sum = 0.0
                for key in model_grad:
                    # calculate sum up values of all absolute gradients
                    sensitivity_abs_sum += torch.sum(torch.abs(model_grad[key]))
                    stop =1
                
                # iterate through all layers again to normalize the absolute gradients from a model-like gradients
                for key in model_grad:
                    sensitivity_norm = torch.div(torch.abs(model_grad[key]), sensitivity_abs_sum)
                    # s_maps_magitude[batch* train_loader.batch_size + batIdx][key] = copy.deepcopy(sensitivity_norm) # DEL, for validation
                    s_maps_mag_sum[key] += sensitivity_norm # accumulate normalized sensitivity in this layer in this batch
        
        jm_batch_time = time.time() - jm_batch_start_t
        smap_batch_times.append(jm_batch_time)
    
    # Finish calculating absolute magnitude of gradients for all batches to the model


    # s_maps_mag_sum is an accumulated values of normalized sensitivity values, take average based on the num_images
    s_maps_mag_avg = {}
    for key in s_maps_mag_sum:
        s_maps_mag_avg[key] = torch.div(s_maps_mag_sum[key], num_images) 

    stop = 1
    smap_time = time.time() - smap_jm_start_t
    return s_maps_mag_avg, smap_batch_times, smap_time


def calculate_smap_jacob(train_loader, batchsize, model, criterion, device):
    s_maps_jacob = [{} for _ in range(len(train_loader.dataset))]
    num_images = 0 # for logging progress of batch
    s_maps_layer_time = [{} for _ in range(len(train_loader.dataset))] # for saving execution time of smap per batch
    jm_batch_times = []
    jm_layer_second_der_times = []

    smap_jm_start_t = time.time()
    for batch, (x, y) in enumerate(train_loader):
        jm_batch_start_t = time.time()
        if batch < len(train_loader): 
            x, y = x.to(device), y.to(device)
            num_images += len(y)
            print(f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)})')
           
            output = model(x)
            loss = criterion(output, y.long())

            jm_layer_second_der_time_dict = {}
            key_count = 1
            # for (param_layer, key) in zip(model.parameters(), model.state_dict()): # causes inconsistent layer name and size
            for name, param_layer in model.named_parameters():
                jm_layer_start_t = time.time()
                # print(f"Executing layer {name}, {key_count}/{len(model.state_dict())}")
                print(f"Executing layer {name}, {key_count}/{len(list(model.named_parameters()))}")
                
                if name != "":
                    print(f"batch #{batch}: {param_layer.shape}; {model.state_dict()[name].shape}")
                        # calcualte the first gradient: loss w.r.t model parameters in a layer
                    print(" calculating first grad")
                    param_layer_grad_func_start_t = time.time()
                    param_layer_grad = torch.autograd.grad(outputs=loss, inputs=param_layer, create_graph=True)[0]
                    param_layer_grad_func_end_t = time.time()
                    param_layer_grad_func_time = param_layer_grad_func_end_t - param_layer_grad_func_start_t
                    print(f" first grad calculated, takes {param_layer_grad_func_time}")

                    
                    weight_jm_jacob = [[] for _ in range(batchsize)]
                    # NEW, jacobian
                    print("   calculating jacob...")
                    jm_layer_second_der_start_t2 = time.time()
                    param_layer_grads_vec = parameters_to_vector(param_layer_grad)
                    # results are close to calculate one by one
                    jm_layer_second_der_fun_t_start = time.time()
                    # logging.basicConfig(level=logging.DEBUG)
                    jms_jocob = torch.autograd.functional.jacobian(
                        lambda x: torch.autograd.grad(param_layer_grads_vec.dot(x), output, create_graph=True)[0],
                        torch.ones_like(param_layer_grads_vec)
                    ) # lamda function: name a small function as x, the arguments and functinos are after :
                    
                    jm_layer_second_der_fun_t_end = time.time()
                    jm_layer_second_der_fun_time = jm_layer_second_der_fun_t_end - jm_layer_second_der_fun_t_start
                    

                    # aa_transpose = aa.transpose(-1,  1) # aa was [batch, #classes, #layer param], transpose to [batch, #layer param, #classes]
                    jms_jocob_transpose = jms_jocob.permute(2, 0, 1) 
                    # separated_tensors = list(torch.unbind(jms_jocob_transpose, dim=0))
                    print(f"   jacob calculated, jacob func time: {jm_layer_second_der_fun_time}")
                    # convert one hot into a single value for jm
                    
                    jm_layer_second_der_end_t2 = time.time()
                    jm_layer_second_der_time2 = jm_layer_second_der_end_t2 - jm_layer_second_der_start_t2  

                    # Sum the absolute values along the last dimension (one-hot dimension)
                    abs_sum_jacob = torch.sum(torch.abs(jms_jocob_transpose), dim=2)
                    # Transpose to get the shape [batch, #layer_param]
                    abs_sum_jacob = abs_sum_jacob.t()
                    # Convert to a list of lists
                    weight_jm_jacob = abs_sum_jacob.tolist()

                    print(f"   jacob processed, takes {jm_layer_second_der_time2}")
                    
                    # reshape the list to tensors (maps that are the same dim as the model)
                    flat_tensor_jacob = torch.tensor([])
                    for idx, each_weight_jm_jacob in enumerate(weight_jm_jacob): # i iterates through 0 ~ batchsize
                        # the last batch may not have the #batch items, null items don't have length
                        if len(each_weight_jm_jacob) > 0: # each_weight_jm: len of the layer
                            flat_tensor_jacob = torch.tensor(each_weight_jm_jacob)
                            # reshaped_tensor = flat_tensor_jacob.reshape(model.state_dict()[key].shape) # causes error "shape '[32]' is invalid for input of size 288" for layer stem.0.bn.running_mean
                            reshaped_tensor = flat_tensor_jacob.reshape(param_layer.shape)

                            # batch: number of current executed batch, use it to control with location to store
                            # print(f"     idx: {batch* batchsize + idx} with key {key}") 
                            s_maps_jacob[batch* batchsize + idx][name] = reshaped_tensor # consumes memory
                            # valid_idx = batch* batchsize + idx # DEL, only for small portion of batches

                jm_layer_end_t = time.time()
                jm_layer_time = jm_layer_end_t - jm_layer_start_t
                print(f"Layer {name} takes {jm_layer_time} seconds")

                s_maps_layer_time[batch][name] = jm_layer_time
                jm_layer_second_der_times.append(jm_layer_second_der_time_dict)
                key_count +=1

        jm_batch_end_t = time.time()
        jm_batch_time = jm_batch_end_t - jm_batch_start_t 
        jm_batch_times.append(jm_batch_time)
        # print(f"Batch {batch} takes {jm_batch_time} seconds")
        stop =1
    
    smap_jm_end_t = time.time()
    smap_jm_time = smap_jm_end_t - smap_jm_start_t
    return num_images, s_maps_jacob, s_maps_layer_time, jm_layer_second_der_times, jm_batch_times, smap_jm_time

def avg_batch_maps(s_maps_jacob, model, num_images):
    # jacob
    smap_calc_avg_start_t = time.time()
    map_sum_jacob = copy.deepcopy(s_maps_jacob[0]) # for calculating averaged smap for #data samples
    map_avg_jacob = {}


    for key in model.state_dict().keys(): # each layer
        if key in s_maps_jacob[0].keys(): 
            for i in range(1, len(s_maps_jacob)): # each batch item
                if bool(s_maps_jacob[i]): #DEL: only records small layers
                    map_sum_jacob[key] = torch.add(torch.abs(map_sum_jacob[key]), torch.abs(s_maps_jacob[i][key]))

            map_avg_jacob[key] = torch.div(map_sum_jacob[key], num_images)
            stop = 1
    
    smap_calc_avg_end_t = time.time()
    smap_calc_avg_time = smap_calc_avg_end_t - smap_calc_avg_start_t
    
    return map_avg_jacob, map_sum_jacob, smap_calc_avg_time 

def log_weights(writer, model, step):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

def visualize_layer_weights(writer, model, step):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy()
            
            # Reshape to 2D
            if len(weights.shape) == 4:  # Convolutional layer
                c_out, c_in, h, w = weights.shape
                weights_2d = weights.reshape(c_out, -1)
            elif len(weights.shape) == 2:  # Linear layer
                weights_2d = weights
            else:
                continue  # Skip other types of layers
            
            # Create a figure and axis
            plt.rcParams['font.size'] = 100
            fig, ax = plt.subplots(figsize=(20, 20))
            
            # Plot the weights
            im = ax.imshow(weights_2d, cmap='hot_r', aspect='auto')
            
            # Add a colorbar
            plt.colorbar(im)
            
            # Set title
            ax.set_title(f'{name} Weights')
            
            # Log to TensorBoard
            writer.add_figure(f'{name}_weights', fig, global_step=step)
            plt.close(fig)


def save_visualized_layer_weights(model, tboard_dir, suffix): # for comparing to jacob
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy()
            
            # Reshape to 2D
            if len(weights.shape) == 4:  # Convolutional layer
                c_out, c_in, h, w = weights.shape
                weights_2d = weights.reshape(c_out, -1)
            elif len(weights.shape) == 2:  # Linear layer
                weights_2d = weights
            else:
                continue  # Skip other types of layers
            
            # Create a figure and axis
            plt.rcParams['font.size'] = 30
            fig, ax = plt.subplots(figsize=(20, 20))
            
            # Plot the weights
            im = ax.imshow(weights_2d, cmap='hot_r', aspect='auto')
            
            # Add a colorbar
            plt.colorbar(im)
            
            # Set title
            # name = name.split('.')[0]
            ax.set_title(f'{name}')
            
            fig.savefig(os.path.join(tboard_dir, f'{name}_s_map_{suffix}.png'))
            plt.close(fig)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 

def sort_and_get_top_params_globally(model, ratio):
    all_weights = []
    weight_info = []
    param_shapes = {}

    # Collect all weights and their info
    for name, param in model.named_parameters():
        # if 'weight' in name:  # Consider only weight parameters
        # Consider both bias and weight
            flat_params = param.data.flatten()
            param_shapes[name] = param.shape

            for idx, p in enumerate(flat_params):
                all_weights.append(p.abs().item()) # Calculate the intensity
                multi_idx = np.unravel_index(idx, param.shape)
                weight_info.append((name, p.item(), multi_idx, idx)) # idx is used for getting the position in flatten models

    # Sort all weights
    sorted_indices = sorted(range(len(all_weights)), key=lambda k: all_weights[k], reverse=True)

    # Calculate how many weights to keep
    num_to_keep = int(len(all_weights) * ratio)

    # Get the top weights
    top_weights = []
    for i in sorted_indices[:num_to_keep]:
        top_weights.append(weight_info[i])

    # Create mask
    mask = {}
    for name, shape in param_shapes.items():
        mask[name] = torch.zeros(shape)

    # Fill mask with 1s for top weights
    for name, _, multi_idx, _ in top_weights:
        mask[name][multi_idx] = 1


    return top_weights, mask


def sort_and_get_top_weights_layer(model, ratio):
    # Dictionary to store weights and their locations
    weights_dict = defaultdict(list)

    # Iterate through named parameters
    for name, param in model.named_parameters():
        if 'weight' in name:  # Consider only weight parameters
            # Flatten the weight tensor
            flat_weights = param.data.flatten()
            
            # Get the indices that would sort the tensor in descending order
            sorted_indices = torch.argsort(flat_weights.abs(), descending=True)
            
            # Calculate how many weights to keep
            num_to_keep = int(len(flat_weights) * ratio)
            
            # Get the top weights and their indices
            top_weights = flat_weights[sorted_indices[:num_to_keep]]
            top_indices = sorted_indices[:num_to_keep]
            
            # Store the results
            for w, idx in zip(top_weights, top_indices):
                # Convert flattened index back to multidimensional index
                multi_idx = np.unravel_index(idx.item(), param.shape)
                weights_dict[name].append((w.item(), multi_idx))

    return weights_dict

def mask_model(model, mask):
    masked_model = {} # for encryption
    unmasked_model = {}
    unmasked_part_model = {} # for not encrypted parameters

    for name, param in model.named_parameters():
        masked_model[name] = param.data * mask[name].to(param.data.device)
        unmasked_model[name] = param.data * (1-mask[name].to(param.data.device))

        # Flatten the parameter and mask
        flat_param = param.data.flatten()
        flat_mask = mask[name].to(param.data.device).flatten()
        
        # Keep only the values where mask is not 0
        unmasked_values = flat_param[flat_mask == 0]
        unmasked_part_model[name] = unmasked_values

        
    return masked_model, unmasked_model, unmasked_part_model # unmasked_model: not used

def recover_ptxt_shape(model, unmasked_part_model_dict, s_mask):
    non_HE_reshape_ptxt_t_start = time.time()
    recovered_model = {}
    for name, param in model.named_parameters():
        device = param.data.device
        original_shape = param.data.shape
        
        # Create a zero tensor with the same shape as the original parameter
        recovered_param = torch.zeros_like(param.data)
        
        # Flatten the mask
        flat_mask = s_mask[name].to(device).flatten()
        flat_mask_bool = (flat_mask == 0) # mask: 1, for encryption; 0, for plaintext
        recover_indices = [idx for idx in range(len(flat_mask_bool)) if flat_mask_bool[idx] == True]
        
        # Use the mask to place the non-zero values in their original positions
        if(len(unmasked_part_model_dict[name]) > 0):
            for rec_idx, ptxt_param in zip(recover_indices, unmasked_part_model_dict[name]):
                recovered_param.flatten()[rec_idx] = ptxt_param
        
        # Reshape to the original shape
        recovered_model[name] = recovered_param.view(original_shape)
    
    non_HE_reshape_ptxt_time = time.time() - non_HE_reshape_ptxt_t_start

    return recovered_model, non_HE_reshape_ptxt_time

def recover_ptxt_shape2(model, unmasked_part_model_dict, s_mask):
    non_HE_reshape_ptxt_t_start = time.time()
    recovered_model = {}
    for name, param in model.named_parameters():
        device = param.data.device
        original_shape = param.data.shape
        
        # Create a zero tensor with the same shape as the original parameter
        recovered_param = torch.zeros_like(param.data)
        
        # Flatten the mask
        flat_mask = s_mask[name].to(device).flatten()
        flat_mask_bool = (flat_mask == 0) # mask: 1, for encryption; 0, for plaintext
        
        # Use the mask to place the non-zero values in their original positions
        if(len(unmasked_part_model_dict[name]) > 0):
            # Use boolean indexing to assign values in one operation
            recovered_param.flatten()[flat_mask_bool] = torch.tensor(unmasked_part_model_dict[name], device=device).clone().detach()

        
        # Reshape to the original shape
        recovered_model[name] = recovered_param.view(original_shape)
    
    non_HE_reshape_ptxt_time = time.time() - non_HE_reshape_ptxt_t_start

    return recovered_model, non_HE_reshape_ptxt_time


def recover_ptxt_shape3(model, unmasked_part_model_dict, s_mask):
    non_HE_reshape_ptxt_t_start = time.time()
    recovered_model = {}
    
    for name, param in model.named_parameters():
        device = param.device
        original_shape = param.shape
        
        # Create a zero tensor with the same shape as the original parameter
        recovered_param = torch.zeros(original_shape, device=device)
        
        # Move mask to device and create boolean mask
        flat_mask = s_mask[name].to(device, non_blocking=True).flatten()
        flat_mask_bool = (flat_mask == 0)  # mask: 1 for encryption; 0 for plaintext
        
        # Use the mask to place the non-zero values in their original positions
        if len(unmasked_part_model_dict[name] > 0):
            unmasked_tensor = torch.tensor(unmasked_part_model_dict[name], device=device).clone().detach()
            recovered_param.flatten()[flat_mask_bool] = unmasked_tensor
        
        recovered_model[name] = recovered_param
    
    non_HE_reshape_ptxt_time = time.time() - non_HE_reshape_ptxt_t_start
    return recovered_model, non_HE_reshape_ptxt_time