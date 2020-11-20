# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:59:03 2020

@author: udits
"""

from __future__ import print_function, division
import os
import torch
#import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import pickle
from VRAIDataset import VRAIDataset, Rescale, VRAIDataset_val
#from model import AttributePrediction
from model_new import AttributePrediction
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter
from test import validate, visualize_grid
import itertools

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')

def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)
	
def calculate_metrics(output, target):
	#print("From Calculate_matrices: ", str(target.shape))
	_, predicted_color = output['color'].cpu().max(1)
	gt_color = target['color_labels'].cpu()
	
	_, predicted_type = output['type'].cpu().max(1)
	gt_type = target['type_labels'].cpu()
	
	_, predicted_wheel = output['wheel'].cpu().max(1)
	gt_wheel = target['wheel_labels'].cpu()
	
	_, predicted_luggage = output['luggage'].cpu().max(1)
	gt_luggage = target['luggage_labels'].cpu()
	
	_, predicted_sky = output['sky'].cpu().max(1)
	gt_sky = target['sky_labels'].cpu()
	
	_, predicted_bumper = output['bumper'].cpu().max(1)
	gt_bumper = target['bumper_labels'].cpu()
	
	with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
		warnings.simplefilter("ignore")
		accuracy_color = accuracy_score(y_true=gt_color.numpy(), y_pred=predicted_color.numpy())
		accuracy_type = accuracy_score(y_true=gt_type.numpy(), y_pred=predicted_type.numpy())
		accuracy_wheel = accuracy_score(y_true=gt_wheel.numpy(), y_pred=predicted_wheel.numpy())
		accuracy_luggage = accuracy_score(y_true=gt_luggage.numpy(), y_pred=predicted_luggage.numpy())
		accuracy_sky = accuracy_score(y_true=gt_sky.numpy(), y_pred=predicted_sky.numpy())
		accuracy_bumper = accuracy_score(y_true=gt_bumper.numpy(), y_pred=predicted_bumper.numpy())
		
	accs_dict = {'type_acc': accuracy_type, 'color_acc': accuracy_color, 
									   'wheel_acc': accuracy_wheel, 'luggage_acc':accuracy_luggage, 
									   'sky_acc': accuracy_sky, 
									   'bumper_acc':accuracy_bumper}
	return accs_dict

def my_collate(batch):
	batch = filter(lambda x: x['labels']['color_labels']!= 0, batch)
	return torch.utils.data.dataloader.default_collate(list(batch))

def plot_confusion_matrix(cm, classes, label,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label'); plt.savefig(('./results/%s_conf_mat.png' % label)); plt.close()
	
def conf_mat_helper(labels,val_split_size):
	temp = []
	for i in range(len(labels)):
		for j in range(len(labels[i])):
			temp.append(labels[i][j])
	return np.array(temp).reshape(val_split_size)

def train(start_epoch=1, N_epochs=20, batch_size=50, num_workers=8):
	device = torch.device('cuda')
	#torch.cuda.device('cuda:1')
	img_size = (256,256)
	train_split_size = 55000
	total_len_dev = 59608
	val_split_size = 11113
	train_transform = transforms.Compose([
		transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
	
	val_transform = transforms.Compose([
        transforms.Resize(img_size),
		transforms.ToTensor()
    ])
	
	train_dataset = VRAIDataset(pickle_file="VRAI/train_annotation.pkl",  image_directory="VRAI/images_train/", 
                                     transform = train_transform)
	#print(len(train_dataset))
	#train_split, val_split = random_split(train_dataset,
     #                                          [train_split_size, val_split_size], generator=torch.Generator().manual_seed(42))
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers)
	
	val_dataset = VRAIDataset_val(pickle_file="VRAI/test_dev_annotation.pkl",  image_directory = "VRAI/images_dev/", 
                                     transform=val_transform)
	val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
	
	model = AttributePrediction(n_color_classes=9,
                             n_type_classes=7, n_wheel_classes = 2, n_luggage_classes = 2, n_sky_classes = 2, n_bumper_classes = 2
                             ).to(device)
							
	optimizer = torch.optim.Adam(model.parameters())
	
	logdir = os.path.join('./logs/', get_cur_time())
	savedir = os.path.join('./checkpoints/', get_cur_time())
	os.makedirs(logdir, exist_ok=True)
	os.makedirs(savedir, exist_ok=True)
	#logger = SummaryWriter(logdir)
	
	n_train_samples = len(train_dataloader)
	n_val_samples = len(val_dataloader)
	print(n_val_samples)
	
	epochs = []
	loss = []
	acc_col = []
	acc_type = []
	acc_wheel = []
	acc_luggage = []
	acc_sky = []
	acc_bumper = []
	metrics_val = []

	print("Starting training ...")
	
	for epoch in range(start_epoch, N_epochs + 1):
		total_loss = 0
		accuracy_color = 0
		accuracy_type = 0
		accuracy_wheel = 0
		accuracy_luggage = 0
		accuracy_sky = 0
		accuracy_bumper = 0
		count = 0
		keys = ['color', 'type', 'wheel', 'luggage', 'sky', 'bumper']
		#d = dict.fromkeys(keys, [])
		predictions = {}
		ground_truths = {}
		#values = [1,2,3,5,6,7,7]
		for i in keys:
			predictions[i] = []
			ground_truths[i] = []
		#preds_color = []
		#gts_color = [] 
		#preds_type = [] 
		#gts_type = []
		for batch in train_dataloader:
			if count % 200 == 0:
				print(str(epoch), "th Epoch, ", str(count), "th batch")
			count += 1
			optimizer.zero_grad()
			
			img = batch['image']
			
			target_labels = batch['labels']
			#print("labels before conversion: ", target_labels)
			target_labels = {t: target_labels[t].to(device) for t in target_labels}
			#print(img.shape)
			#print(target_labels)
			output = model(img.to(device))
			
			loss_train, losses_train = model.get_loss(output, target_labels)
			#print(loss_train.item())
			total_loss += loss_train.item()
			#total_loss += loss_train
			batch_accuracy_dict = calculate_metrics(output, target_labels)
			
			accuracy_color += batch_accuracy_dict['color_acc']
			accuracy_type += batch_accuracy_dict['type_acc']
			accuracy_wheel += batch_accuracy_dict['wheel_acc']
			accuracy_luggage += batch_accuracy_dict['luggage_acc']
			accuracy_sky += batch_accuracy_dict['sky_acc']
			accuracy_bumper += batch_accuracy_dict['bumper_acc']
 			
			loss_train.backward()
			optimizer.step()
			
		print("epoch {:4d}, loss: {:.4f}, color: {:.4f}, type: {:.4f}, wheel: {:.4f}, luggage: {:.4f}, sky: {:.4f}, bumper: {:.4f}".format(
            epoch, total_loss / n_train_samples, accuracy_color / n_train_samples, accuracy_type / n_train_samples,
			accuracy_wheel / n_train_samples, accuracy_luggage / n_train_samples, accuracy_sky / n_train_samples,
			accuracy_bumper / n_train_samples))
		
		#logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)
		
		epochs.append(epoch)
		
		acc_col.append(accuracy_color/n_train_samples)
		acc_type.append(accuracy_type / n_train_samples)
		acc_wheel.append(accuracy_wheel / n_train_samples)
		acc_luggage.append(accuracy_luggage / n_train_samples)
		acc_sky.append(accuracy_sky / n_train_samples)
		acc_bumper.append(accuracy_bumper / n_train_samples)
		
		loss.append(total_loss / n_train_samples)
		
		if epoch % 2 == 0:
			met_val, predictions, ground_truths = validate(model, val_dataloader, epoch, device, predictions, ground_truths)
			metrics_val.append(met_val)
		if epoch % 10 == 0:
			checkpoint_save(model, savedir, epoch)
			
	checkpoint_path = checkpoint_save(model, savedir, epoch - 1)
	metrics = np.array((epochs, acc_col,acc_type,loss))
	metrics_val_np = np.array(metrics_val)
	
	gts_color = conf_mat_helper(ground_truths['color'],total_len_dev)
	preds_color = conf_mat_helper(predictions['color'],total_len_dev)
		
	gts_type = conf_mat_helper(ground_truths['type'],total_len_dev)
	preds_type = conf_mat_helper(predictions['type'],total_len_dev)
	
	gts_wheel = conf_mat_helper(ground_truths['wheel'],total_len_dev)
	preds_wheel = conf_mat_helper(predictions['wheel'],total_len_dev)
	
	gts_luggage = conf_mat_helper(ground_truths['luggage'],total_len_dev)
	preds_luggage = conf_mat_helper(predictions['luggage'],total_len_dev)
	
	gts_sky = conf_mat_helper(ground_truths['sky'],total_len_dev)	
	preds_sky = conf_mat_helper(predictions['sky'],total_len_dev)
	
	gts_bumper = conf_mat_helper(ground_truths['bumper'],total_len_dev)
	preds_bumper = conf_mat_helper(predictions['bumper'],total_len_dev)
	
	#Plot confusion matrix
	cn_matrix_color = confusion_matrix(
            y_true=gts_color,
            y_pred=preds_color)
	
	cn_matrix_type = confusion_matrix(
            y_true=gts_type,
            y_pred=preds_type)
	
	cn_matrix_luggage = confusion_matrix(
            y_true=gts_luggage,
            y_pred=preds_luggage)
	cn_matrix_wheel = confusion_matrix(
            y_true=gts_wheel,
            y_pred=preds_wheel)
	cn_matrix_sky = confusion_matrix(
            y_true=gts_sky,
            y_pred=preds_sky)
	cn_matrix_bumper = confusion_matrix(
            y_true=gts_bumper,
            y_pred=preds_bumper)
	
	#print("Confusion matrix for color label: ", cn_matrix_color)
	#print("Confusion matrix for type label: ", cn_matrix_type)
	np.save('./results/train_metrics_20e_dropout_03_e_same.npy', metrics)
	np.save('./results/val_metrics_20e_dropout_03_e_same.npy', metrics_val_np)
	#visualize_grid(model, val_dataloader, attributes, device, checkpoint=checkpoint_path)
	
	cm_color_labels = ['White', 'Black', 'Gray', 'Red', 'Green', 'Blue', 'Yellow', 'Brown', 'Other']
	
	cm_type_labels = ['Sedan', 'Hatchback', 'SUV', 'Bus', 'Lorry', 'Truck', 'Other']
	
	cm_binary_labels = ['0', '1']
	
	print()
	
	plot_confusion_matrix(cm=cn_matrix_color, classes=cm_color_labels, label = 'color', title='Confusion Matrix for Color label')
	
	plot_confusion_matrix(cm=cn_matrix_type, classes=cm_type_labels, label = 'type', title='Confusion Matrix for Type label', cmap = plt.cm.Reds)
	
	plot_confusion_matrix(cm=cn_matrix_luggage, classes=cm_binary_labels, label = 'luggage', title='Confusion Matrix for Luggage label', cmap = plt.cm.Greys)
	
	plot_confusion_matrix(cm=cn_matrix_wheel, classes=cm_binary_labels, label = 'wheel', title='Confusion Matrix for Wheel label', cmap = plt.cm.Purples)
	
	plot_confusion_matrix(cm=cn_matrix_sky, classes=cm_binary_labels, label = 'sky', title='Confusion Matrix for Sky label', cmap = plt.cm.Greens)
	
	plot_confusion_matrix(cm=cn_matrix_bumper, classes=cm_binary_labels, label = 'bumper', title='Confusion Matrix for Bumper label', cmap = plt.cm.Oranges)

	return checkpoint_path
		
last_checkpoint_path = train()

	