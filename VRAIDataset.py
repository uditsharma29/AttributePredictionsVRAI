# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:15:48 2020

@author: udits
"""
from __future__ import print_function, division
import os
import torch
#import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from PIL import Image
import numpy as np

class VRAIDataset(Dataset):
	#Dataset builder for the VRAI dataset
	def __init__(self, pickle_file, image_directory, transform=None):
		self.pickle_file = pickle.load( open(pickle_file, "rb" ))
		self.transform = transform
		#self.drive = "C:/"
		self.directory = image_directory
		
	def __len__(self):
		return len(self.pickle_file['train_im_names'])
		#return 1000
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		directory = self.directory
		pickle_file = self.pickle_file
		#images = []
		#labels = []
		
		image_name = directory+pickle_file['train_im_names'][idx]
		#print(image_name)
		image = Image.open(image_name)
		#print(image.shape)
		if self.transform:
 			image = self.transform(image)
		
		ID = int(pickle_file['train_im_names'][idx].split('_')[0])
		
		wheel_label = pickle_file['wheel_label'][ID]
		type_label = pickle_file['type_label'][ID]
		color_label = pickle_file['color_label'][ID]
		luggage_label = pickle_file['luggage_label'][ID]
		sky_label = pickle_file['sky_label'][ID]
		bumper_label = pickle_file['bumper_label'][ID]
		
		#labels = np.array([type_label, color_label])
		
		#print(labels)
# 		if color_label == 0:
# 			return None
# 		else:
		sample = {'image': image, 'labels': {'type_labels': type_label, 'color_labels': color_label, 
									   'wheel_labels': wheel_label, 'luggage_labels': luggage_label, 'sky_labels': sky_label, 
									   'bumper_labels':bumper_label}}	
		return sample
	
class VRAIDataset_val(Dataset):
	#Dataset builder for the VRAI dataset
	def __init__(self, pickle_file, image_directory, transform=None):
		self.pickle_file = pickle.load( open(pickle_file, "rb" ))
		self.transform = transform
		#self.drive = "C:/"
		self.directory = image_directory
		
	def __len__(self):
		return len(self.pickle_file['dev_im_names'])
		#return 1000
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		directory = self.directory
		pickle_file = self.pickle_file
		#images = []
		#labels = []
		
		image_name = directory+pickle_file['dev_im_names'][idx]
		#print(image_name)
		key = pickle_file['dev_im_names'][idx]
		
		image = Image.open(image_name)
		#print(image.shape)
		if self.transform:
 			image = self.transform(image)
		
		#ID = int(pickle_file['train_im_names'][idx].split('_')[0])
		
		wheel_label = pickle_file['wheel_label'][key]
		type_label = pickle_file['type_label'][key]
		color_label = pickle_file['color_label'][key]
		luggage_label = pickle_file['luggage_label'][key]
		sky_label = pickle_file['sky_label'][key]
		bumper_label = pickle_file['bumper_label'][key]
		
		#labels = np.array([type_label, color_label])
		
		#print(labels)
# 		if color_label == 0:
# 			return None
# 		else:
		sample = {'image': image, 'labels': {'type_labels': type_label, 'color_labels': color_label, 
									   'wheel_labels': wheel_label, 'luggage_labels': luggage_label, 'sky_labels': sky_label, 
									   'bumper_labels':bumper_label}}	
		return sample
	
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}
	
class Rescale(object):
	def __init__(self, output_h, output_w):
        #assert isinstance(output_size, (int, tuple))
		self.output_h = output_h
		self.output_w = output_w
		
	def __call__(self, sample):
		image, labels = sample['image'], sample['labels']
		h, w = image.shape[:2]
		#if isinstance(self.output_size, int):
           # if h > w:
            #    new_h, new_w = self.output_size * h / w, self.output_size
            #else:
		new_h, new_w = self.output_h, self.output_w
		#else:
		#	new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]
		return {'image': img, 'labels': labels}
	
# vrai_dataset = VRAIDataset(pickle_file="C:/MS/Thesis/VRAI Dataset/VRAI-20200818T021818Z-002/VRAI/train_annotation.pkl",
#                                     root_dir='')

# for i in range(len(vrai_dataset)):
# 	sample = vrai_dataset[i]
# 	print(i, sample['image'].shape, sample['labels'].shape)
# 	print(sample['labels'])
# 	break

# vrai_dataset_transformed = VRAIDataset(pickle_file="C:/MS/Thesis/VRAI Dataset/VRAI-20200818T021818Z-002/VRAI/train_annotation.pkl",
#                                      transform=transforms.Compose([Rescale(300, 200),                                             
#                                                ToTensor()
#                                            ]))

# dataloader = DataLoader(vrai_dataset_transformed, batch_size=4,
#                         shuffle=False, num_workers=0)

# for i_batch, sample_batched in enumerate(dataloader):
# 	print(i_batch, sample_batched['image'].size(), sample_batched['labels'].size())
# 	break
# 	