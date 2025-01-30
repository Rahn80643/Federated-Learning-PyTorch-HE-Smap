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

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets as tvdatasets, transforms as tvtransforms
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
import models
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import inspect
# from . import TinyImageNetDataset


class TinyImageNetDataset(Dataset): # For getting the entire dataset before splitting into clients
    def __init__(self, root_dir, train=True, transform=None, augment=None, normalize=None, name=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.to_tensor = tvtransforms.PILToTensor()
        
        # Define classes
        self.classes = sorted(os.listdir(os.path.join(root_dir, 'train')))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

       
        self.augment = augment
        self.normalize = normalize
        
        def load_image(img_path):
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            return img_bytes
        
        if self.train:
            # Count total number of training images
            num_samples = sum([len(os.listdir(os.path.join(root_dir, 'train', cls, 'images'))) for cls in self.classes])
            
            # Pre-allocate numpy array for image bytes
            self.data = np.empty(num_samples, dtype=object)
            self.targets = np.empty(num_samples, dtype=int)

            # Load all training images
            idx = 0
            for label, cls in enumerate(tqdm(self.classes, desc="Loading training images")):
                cls_dir = os.path.join(root_dir, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    # img_bytes = load_image(img_path)
                    self.data[idx] = Image.open(img_path).convert('RGB')
                    self.targets[idx] = label
                    idx += 1
        else:
            val_dir = os.path.join(root_dir, 'val', 'images')
            val_annotations = pd.read_csv(os.path.join(root_dir, 'val', 'val_annotations.txt'),
                                          sep='\t', header=None,
                                          names=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'])
            
            # Pre-allocate numpy array for image bytes
            num_samples = len(val_annotations)
            self.data = np.empty(num_samples, dtype=object)
            self.targets = np.empty(num_samples, dtype=int)

            # Load all validation images
            for idx, row in tqdm(val_annotations.iterrows(), total=num_samples, desc="Loading validation images"):
                # img_bytes = load_image(os.path.join(val_dir, row['file_name']))
                self.data[idx] = Image.open(os.path.join(val_dir, row['file_name'])).convert('RGB')
                self.targets[idx] = self.class_to_idx[row['class']]

        print(f"Loaded {len(self.data)} images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, augmented=True, normalized=True):
        image = self.data[idx]
        label = self.targets[idx]
        
        # Convert bytes to PIL Image
        # image = Image.open(io.BytesIO(img_bytes))        
        # Convert PIL Image to tensor
        image = self.to_tensor(image)
        # print(f"In TinyImageNetDataset, image range: [{image.min()}, {image.max()}]")

        return image, label

class TinyImageNetSubset(TinyImageNetDataset): 
    def __init__(self, dataset, idxs, augment=None, normalize=None, name=None):
        self.name = name if name is not None else dataset.name
        self.dataset = dataset.dataset if 'dataset' in vars(dataset) else dataset
        self.idxs = idxs
        self.targets = np.array(dataset.targets)[idxs]
        self.augment = augment
        
        # Copy classes and class_to_idx from the original dataset
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

        if augment is None:
            self.augment = dataset.augment if 'augment' in vars(dataset) else None
        else:
            self.augment = augment

        if normalize is None:
            self.normalize = dataset.normalize if 'normalize' in vars(dataset) else None
        else:
            self.normalize = normalize

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx, augmented=True, normalized=True):
        # # debug
        # # Get the caller's frame
        # caller_frame = inspect.currentframe().f_back
        
        # # Get caller function name
        # caller_name = caller_frame.f_code.co_name
        
        # # Get caller file name and line number
        # caller_filename = caller_frame.f_code.co_filename
        # caller_line = caller_frame.f_lineno
        
        # print(f"Called by {caller_name} from {caller_filename} at line {caller_line}")

        # # debug
        
        image, label = self.dataset[self.idxs[idx]]

        # print(f"TinyImageNetSubset, self.dataset: {type(self.dataset)} type of image: {image.type}; idx: {idx}")

        # convert int [0, 255] to float [0.0, 1.0]
        image = image.float() / 255.0
        if normalized and self.normalize is not None:
            image = self.normalize(image)
        
        if augmented and self.augment is not None:
            image = self.augment(image)        

        return image, label
    
    def __str__(self):
        dataset_str = f'Name: {self.name}\n'\
                      f'Number of samples: {len(self)}\n'\
                      f'Class distribution: {[(self.targets == c).sum() for c in range(len(self.classes))]}\n'\
                      f'Augmentation: {self.augment}\n'\
                      f'Normalization: {self.normalize}'
        return dataset_str


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs, augment=None, normalize=None, name=None):
        self.name = name if name is not None else dataset.name
        self.dataset = dataset.dataset if 'dataset' in vars(dataset) else dataset
        self.idxs = idxs
        self.targets = np.array(dataset.targets)[idxs]
        self.classes = dataset.classes

        if augment is None:
            self.augment = dataset.augment if 'augment' in vars(dataset) else None
        else:
            self.augment = augment

        if normalize is None:
            self.normalize = dataset.normalize if 'normalize' in vars(dataset) else None
        else:
            self.normalize = normalize

        stop =1

    def __getitem__(self, idx, augmented=True, normalized=True):

        # # debug
        # # Get the caller's frame
        # caller_frame = inspect.currentframe().f_back
        
        # # Get caller function name
        # caller_name = caller_frame.f_code.co_name
        
        # # Get caller file name and line number
        # caller_filename = caller_frame.f_code.co_filename
        # caller_line = caller_frame.f_lineno
        
        # print(f"Called by {caller_name} from {caller_filename} at line {caller_line}")
        # debug
        
        example, target = self.dataset[self.idxs[idx]]
        example = tvtransforms.ToTensor()(example)
        # print(f"In Subset, image range: [{example.min()}, {example.max()}]")

        if augmented and self.augment is not None:
            example = self.augment(example)
        if normalized and self.normalize is not None:
            example = self.normalize(example)
        # print(f"In Subset, normalized image range: [{example.min()}, {example.max()}]")

        # #  debug
        # mean = torch.tensor(self.normalize.mean, dtype=torch.float32)
        # std = torch.tensor(self.normalize.std, dtype=torch.float32)
        # denormalize = tvtransforms.Normalize(
        #     mean=[-m/s for m, s in zip(mean, std)],
        #     std=[1/s for s in std]
        # ) # https://github.com/pytorch/vision/issues/528
        # image_inv = denormalize(example)
        # print(f"In Subset, normalize_inv image range: [{image_inv.min()}, {image_inv.max()}]")
        
        return example, target

    def __len__(self):
        return len(self.targets)

    def __str__(self):
        dataset_str = f'Name: {self.name}\n'\
                      f'Number of samples: {len(self)}\n'\
                      f'Class distribution: {[(self.targets == c).sum() for c in range(len(self.classes))]}\n'\
                      f'Augmentation: {self.augment}\n'\
                      f'Normalization: {self.normalize}'
        return dataset_str

def get_mean_std(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    mean = 0.
    var = 0.

    for examples, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        examples = examples.view(examples.size(0), examples.size(1), -1)
        # Update total number of images
        total += examples.size(0)
        # Compute mean and var here
        mean += examples.mean(2).sum(0)
        var += examples.var(2).sum(0)

    # Final step
    mean /= total
    var /= total

    return mean.tolist(), torch.sqrt(var).tolist()

def get_datasets(name, train_augment, test_augment, args):
    train_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=True, download=True)
    test_tvdataset = getattr(tvdatasets, name)(root='data/'+name, train=False, download=False)

    # Determine training, validation and test indices
    if args.frac_valid > 0:
        train_idxs, valid_idxs = train_test_split(range(len(train_tvdataset)), test_size=args.frac_valid, stratify=train_tvdataset.targets)
    else:
        train_idxs, valid_idxs = range(len(train_tvdataset)), None
    test_idxs = range(len(test_tvdataset))

    # Create training, validation and test datasets
    train_dataset = Subset(dataset=train_tvdataset, idxs=train_idxs, augment=train_augment, name=name)
    valid_dataset = Subset(dataset=train_tvdataset, idxs=valid_idxs, augment=test_augment, name=name) if valid_idxs is not None else None
    test_dataset = Subset(dataset=test_tvdataset, idxs=test_idxs, augment=test_augment, name=name)

    if "mnist" in args.dataset:
        mean, std = 0.1307, 0.3081 # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457

    else:
        # Normalization based on pretraining or on previous transforms
        if 'pretrained' in args.model_args and args.model_args['pretrained']:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            # mean, std = get_mean_std(train_dataset, args.test_bs)
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalize = tvtransforms.Normalize(mean, std)
    train_dataset.normalize = normalize
    if valid_dataset is not None: valid_dataset.normalize = normalize
    test_dataset.normalize = normalize

    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}


def get_datasets_tiny_imagenet(name, train_augment, test_augment, args):
    # getting the entire dataset before splitting into clients
    train_tvdataset = TinyImageNetDataset(root_dir='data/tiny-imagenet-200', train=True)
    test_tvdataset = TinyImageNetDataset(root_dir='data/tiny-imagenet-200', train=False)

    # Determine training, validation and test indices
    if args.frac_valid > 0:
        train_idxs, valid_idxs = train_test_split(range(len(train_tvdataset)), test_size=args.frac_valid, stratify=train_tvdataset.targets)
    else:
        train_idxs, valid_idxs = range(len(train_tvdataset)), None
    test_idxs = range(len(test_tvdataset))

    # Create training, validation and test datasets
    train_dataset = TinyImageNetSubset(dataset=train_tvdataset, idxs=train_idxs, augment=train_augment, name=name)
    valid_dataset = TinyImageNetSubset(dataset=train_tvdataset, idxs=valid_idxs, augment=test_augment, name=name) if valid_idxs is not None else None
    test_dataset = TinyImageNetSubset(dataset=test_tvdataset, idxs=test_idxs, augment=test_augment, name=name)

    # Normalization based on pretraining or on previous transforms
    if 'pretrained' in args.model_args and args.model_args['pretrained']:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        # mean, std = get_mean_std(train_dataset, args.test_bs)
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = tvtransforms.Normalize(mean, std) # pass normalization object to dataset loader
    train_dataset.normalize = normalize
    if valid_dataset is not None: valid_dataset.normalize = normalize
    test_dataset.normalize = normalize

    return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}


def get_datasets_fig(datasets, num_examples):
    types, titles = [], []
    for type in datasets:
        if datasets[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    fig, ax = plt.subplots(2, len(types))

    for i, type in enumerate(types):
        examples_orig, examples_trans = [], []
        for idx in torch.randperm(len(datasets[type]))[:num_examples]:
            examples_orig.append(datasets[type].__getitem__(idx, augmented=False, normalized=False)[0])
            examples_trans.append(datasets[type].__getitem__(idx, augmented=True, normalized=False)[0])
        examples_orig = torch.stack(examples_orig)
        examples_trans = torch.stack(examples_trans)

        grid_orig = np.transpose(make_grid(examples_orig, nrow=int(num_examples**0.5)).numpy(), (1,2,0))
        grid_trans = np.transpose(make_grid(examples_trans, nrow=int(num_examples**0.5)).numpy(), (1,2,0))

        ax[0, i].imshow(grid_orig)
        ax[0, i].set_title(titles[i] + ' original')
        ax[1, i].imshow(grid_trans)
        ax[1, i].set_title(titles[i] + ' transformed')

    fig.tight_layout()
    fig.set_size_inches(4*len(types), 8)

    return fig


def get_datasets_fig_tiny_imagenet(datasets, num_examples):
    types, titles = [], []
    for type in datasets:
        if datasets[type] is not None:
            types.append(type)
            titles.append(type.capitalize())
    fig, ax = plt.subplots(2, len(types))

    for i, type in enumerate(types):
        print(f"get_datasets_fig_tiny_imagenet, type: {type}")
        examples_orig, examples_trans = [], []
        for idx in torch.randperm(len(datasets[type]))[:num_examples]:
            example_orig = datasets[type].__getitem__(idx, augmented=False, normalized=False)[0] # denormalize images and convert pixels to integers for visualization
            examples_orig.append(example_orig)

            example_trans = datasets[type].__getitem__(idx, augmented=True, normalized=False)[0]
            examples_trans.append(example_trans)

            # examples_trans.append(datasets[type].__getitem__(idx, augmented=True, normalized=False)[0])
        examples_orig = torch.stack(examples_orig)
        examples_trans = torch.stack(examples_trans)

        grid_orig = np.transpose(make_grid(examples_orig, nrow=int(num_examples**0.5)).numpy(), (1,2,0))
        grid_trans = np.transpose(make_grid(examples_trans, nrow=int(num_examples**0.5)).numpy(), (1,2,0))


        ax[0, i].imshow(grid_orig)
        ax[0, i].set_title(titles[i] + ' original')
        ax[1, i].imshow(grid_trans)
        ax[1, i].set_title(titles[i] + ' transformed')

    fig.tight_layout()
    fig.set_size_inches(4*len(types), 8)

    return fig
