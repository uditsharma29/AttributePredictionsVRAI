# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:00:27 2020

@author: udits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttributePrediction(nn.Module):
	def __init__(self, n_color_classes, n_type_classes, n_wheel_classes, n_luggage_classes, n_sky_classes, n_bumper_classes):
		super().__init__()
		self.base_model = models.resnet101(pretrained=True)  # take the model without classifier
		modules = list(self.base_model.children())[:-1]      # delete the last fc layer.
		self.base_model = nn.Sequential(*modules)
		#last_channel = models.resnet101().last_channel  # size of the layer before classifier
		#num_features = self.base_model.fc.in_features

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		
		self.pool1 = nn.MaxPool2d(3, stride=2)
		
		self.softmax = nn.Softmax(dim=1)
		
		self.fc1 = nn.Linear(2048, 1024)
		self.fc2 = nn.Linear(2048, 512)
		self.fc3 = nn.Linear(1024,512)
		self.fc4 = nn.Linear(512, 256)
		self.fc5 = nn.Linear(512, 128)
		self.fc9 = nn.Linear(256,128)
		self.fc6 = nn.Linear(256, out_features = n_color_classes)
		self.fc7 = nn.Linear(128, out_features = n_color_classes)
		self.fc7 = nn.Linear(128, out_features = n_color_classes)
		
        # create separate classifiers for our outputs
		self.color = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=2048, out_features = n_color_classes)
# 			nn.Linear(in_features=512, out_features=256),
# 			nn.Linear(in_features=256, out_features=n_color_classes)
        )
		self.types = nn.Sequential(
            nn.Dropout(p=0.4),
			#nn.Linear(2048, 512),	
			#nn.ReLU(),
            nn.Linear(in_features=2048, out_features=n_type_classes)
# 			nn.Linear(in_features=512, out_features=256),
# 			nn.Linear(in_features=256, out_featuress=n_type_classes)
        )
		self.wheel = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features = n_wheel_classes)
# 			nn.Linear(in_features=512, out_features=256),
# 			nn.Linear(in_features=256, out_features=n_color_classes)
        )
		self.luggage = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features = n_luggage_classes)
# 			nn.Linear(in_features=512, out_features=256),
# 			nn.Linear(in_features=256, out_features=n_color_classes)
        )
		
		self.sky = nn.Sequential(
            nn.Dropout(p=0.2),
			nn.Linear(2048, 512),
			nn.ReLU(),
            nn.Linear(in_features=512, out_features = n_sky_classes)
# 			nn.Linear(in_features=512, out_features=256),
# 			nn.Linear(in_features=256, out_features=n_color_classes)
        )
		
		self.bumper = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=2048, out_features = n_bumper_classes)
# 			nn.Linear(in_features=512, out_features=256),
# 			nn.Linear(in_features=256, out_features=n_color_classes)
        )
		
	def forward(self, x):
		x = self.base_model(x)
		#print("Shape after RESNET: ", x.shape)
		#x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
		x = torch.flatten(x, 1)	
		
		#type_branch = self.fc1(x)
		#type_branch = self.fc3(type_branch)
		#type_branch = self.fc4(type_branch)
		#type_branch = self.fc9(type_branch)
		
		types  = self.types(x)
		color = self.color(x)
		#wheel = self.wheel(x)
		#luggage = self.luggage(x)
		#sky = self.sky(x)
		#bumper = self.bumper(x)
		
		#types = self.softmax(types)
		#color = self.softmax(color)
		
		return {
            'color': color,
            'type': types,
			#'wheel': wheel,
			#'luggage': luggage,
			#'sky': sky,
			#'bumper': bumper
        }
	def get_loss(self, net_output, ground_truth):
		#print("Ground truth: ",  ground_truth)
		#print("Predictions : ", net_output)
		color_loss = F.cross_entropy(net_output['color'], ground_truth['color_labels'])
		type_loss = F.cross_entropy(net_output['type'], ground_truth['type_labels'])
		wheel_loss = F.binary_cross_entropy(net_output['wheel'], ground_truth['wheel_labels'])
		luggage_loss = F.binary_cross_entropy(net_output['luggage'], ground_truth['luggage_labels'])
		sky_loss = F.binary_cross_entropy(net_output['sky'], ground_truth['type_labels'])
		bumper_loss = F.binary_cross_entropy(net_output['bumper'], ground_truth['bumper_labels'])
		#print(color_loss)
		loss = color_loss + (3*type_loss/2)
		#print(loss)
		return loss, {'color': color_loss, 'type': type_loss}


