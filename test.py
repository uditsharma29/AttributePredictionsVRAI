# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:29:46 2020

@author: udits
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
#from dataset import FashionDataset, AttributesDataset, mean, std
from model import AttributePrediction
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

def validate(model, dataloader, iteration, device, predictions, ground_truths, checkpoint=None):
	if checkpoint is not None:
		checkpoint_load(model, checkpoint)
	
	model.eval()
	with torch.no_grad():
		avg_loss = 0
		accuracy_color = 0
		accuracy_type = 0
		accuracy_wheel = 0
		accuracy_luggage = 0
		accuracy_sky = 0
		accuracy_bumper = 0
		epochs = []
		loss = []
		acc_col = []
		acc_type = []
		acc_wheel = []
		acc_luggage = []
		acc_sky = []
		acc_bumper = []
		for batch in dataloader:
			img = batch['image']
			target_labels = batch['labels']
			target_labels = {t: target_labels[t].to(device) for t in target_labels}
			output = model(img.to(device))
			
			val_train, val_train_losses = model.get_loss(output, target_labels)
			avg_loss += val_train.item()
			accuracies, predictions, ground_truths = calculate_metrics(output, target_labels, predictions, ground_truths)
				
			accuracy_color += accuracies['color']
			accuracy_type += accuracies['type']
			accuracy_wheel += accuracies['wheel']
			accuracy_luggage += accuracies['luggage']
			accuracy_sky += accuracies['sky']
			accuracy_bumper += accuracies['bumper']
	
	n_samples = len(dataloader)
	avg_loss /= n_samples
	accuracy_color /= n_samples
	accuracy_type /= n_samples
	accuracy_wheel /= n_samples
	accuracy_luggage /= n_samples
	accuracy_sky /= n_samples
	accuracy_bumper /= n_samples
	
	print('-' * 72)
	print("Validation  loss: {:.4f}, color: {:.4f}, type: {:.4f}, wheel: {:.4f}, luggage: {:.4f}, sky: {:.4f}, bumper: {:.4f}\n".format(
        avg_loss, accuracy_color, accuracy_type, accuracy_wheel, accuracy_luggage, accuracy_sky, accuracy_bumper))
	epochs.append(iteration)
	acc_col.append(accuracy_color)
	acc_type.append(accuracy_type)
	acc_wheel.append(accuracy_wheel)
	acc_luggage.append(accuracy_luggage)
	acc_sky.append(accuracy_sky)
	acc_bumper.append(accuracy_bumper)
	loss.append(avg_loss)
	
	metrics = np.array((epochs, acc_col,acc_type, acc_wheel, acc_luggage, acc_sky, acc_bumper, loss))
	#logger.add_scalar('val_loss', avg_loss, iteration)
	#logger.add_scalar('val_accuracy_color', accuracy_color, iteration)
	#logger.add_scalar('val_accuracy_type', accuracy_type, iteration)
	
	model.train()
	
	return metrics, predictions, ground_truths

def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_color_all = []
    gt_type_all = []
    predicted_color_all = []
    predicted_type_all = []

    accuracy_color = 0
    accuracy_type = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_colors = batch['labels']['color_labels']
            gt_types = batch['labels']['type_labels']
            output = model(img.to(device))

            batch_accuracy_color, batch_accuracy_type = \
                calculate_metrics(output, batch['labels'])
            accuracy_color += batch_accuracy_color
            accuracy_type += batch_accuracy_type

            # get the most confident prediction for each image
            _, predicted_colors = output['color'].cpu().max(1)
            _, predicted_types = output['type'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_color = attributes.color_id_to_name[predicted_colors[i].item()]
                predicted_type = attributes.type_id_to_name[predicted_types[i].item()]

                gt_color = attributes.color_id_to_name[gt_colors[i].item()]
                gt_types = attributes.type_id_to_name[gt_types[i].item()]

                gt_color_all.append(gt_color)
                gt_type_all.append(gt_types)

                predicted_color_all.append(predicted_color)
                predicted_type_all.append(predicted_type)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(predicted_type, predicted_color))
                gt_labels.append("{}\n{}\n{}".format(gt_types, gt_color))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\ncolor: {:.4f}, type: {:.4f}, article: {:.4f}".format(
            accuracy_color / n_samples,
            accuracy_type / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # color
        cn_matrix = confusion_matrix(
            y_true=gt_color_all,
            y_pred=predicted_color_all,
            labels=attributes.color_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.color_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Colors")
        plt.tight_layout()
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_type_all,
            y_pred=predicted_type_all,
            labels=attributes.type_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.gender_labels).plot(
            xticks_rotation='horizontal')
        plt.title("Genders")
        plt.tight_layout()
        plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()


def calculate_metrics(output, target, predictions, ground_truths):
	_, predicted_color = output['color'].cpu().max(1)
	gt_color = target['color_labels'].cpu()
	
	_, predicted_types = output['type'].cpu().max(1)
	gt_type = target['type_labels'].cpu()
	
	_, predicted_luggage = output['luggage'].cpu().max(1)
	gt_luggage = target['luggage_labels'].cpu()
	
	_, predicted_wheel = output['wheel'].cpu().max(1)
	gt_wheel = target['wheel_labels'].cpu()
	
	_, predicted_sky = output['sky'].cpu().max(1)
	gt_sky = target['sky_labels'].cpu()
	
	_, predicted_bumper = output['bumper'].cpu().max(1)
	gt_bumper = target['bumper_labels'].cpu()
	
	#print(gt_color)
	
	predictions['color'].append(predicted_color.numpy())
	ground_truths['color'].append(gt_color.numpy())
	
	predictions['type'].append(predicted_types.numpy())
	ground_truths['type'].append(gt_type.numpy())
	
	predictions['wheel'].append(predicted_wheel.numpy())
	ground_truths['wheel'].append(gt_wheel.numpy())
	
	predictions['luggage'].append(predicted_luggage.numpy())
	ground_truths['luggage'].append(gt_luggage.numpy())
	
	predictions['sky'].append(predicted_sky.numpy())
	ground_truths['sky'].append(gt_sky.numpy())
	
	predictions['bumper'].append(predicted_bumper.numpy())
	ground_truths['bumper'].append(gt_bumper.numpy())
	
	with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
		warnings.simplefilter("ignore")
		accuracy_color = balanced_accuracy_score(y_true=gt_color.numpy(), y_pred=predicted_color.numpy())
		accuracy_type = balanced_accuracy_score(y_true=gt_type.numpy(), y_pred=predicted_types.numpy())
		accuracy_luggage = balanced_accuracy_score(y_true=gt_luggage.numpy(), y_pred=predicted_luggage.numpy())
		accuracy_wheel = balanced_accuracy_score(y_true=gt_wheel.numpy(), y_pred=predicted_wheel.numpy())
		accuracy_sky = balanced_accuracy_score(y_true=gt_sky.numpy(), y_pred=predicted_sky.numpy())
		accuracy_bumper = balanced_accuracy_score(y_true=gt_bumper.numpy(), y_pred=predicted_bumper.numpy())
		
	accuracies = {'color': accuracy_color, 'type': accuracy_type, 'luggage':accuracy_luggage, 'wheel': accuracy_wheel, 'sky':accuracy_sky, 'bumper': accuracy_bumper}
	
		
	return accuracies, predictions, ground_truths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./fashion-product-images/styles.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_dataset = FashionDataset('./val.csv', attributes, val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    model = MultiOutputModel(n_color_classes=attributes.num_colors, n_gender_classes=attributes.num_genders,
                             n_article_classes=attributes.num_articles).to(device)

    # Visualization of the trained model
    visualize_grid(model, test_dataloader, attributes, device, checkpoint=args.checkpoint)
