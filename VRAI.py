# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:48:48 2020

@author: udits
"""
#%matplotlib inline
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

#from skimage import io, transform
#from VRAIDataset import VRAIDataset

train_annotation = pickle.load( open("C:/MS/Thesis/VRAI Dataset/VRAI-20200818T021818Z-002/VRAI/train_annotation.pkl", "rb" ))
test_annotation = pickle.load( open("C:/MS/Thesis/VRAI Dataset/VRAI-20200818T021818Z-002/VRAI/test_annotation.pkl", "rb" ))

drive = "C:/"
directory = "MS/Thesis/VRAI Dataset/VRAI-20200818T021818Z-002/VRAI"

dict_keys = (['train_im_names', 'wheel_label', 'type_label', 'color_label', 'luggage_label', 'd_part_label', 'sky_label', 'bumper_label'])

color_labels = []
type_labels =[]
wheel_labels = []
sky_labels = []
luggage_labels = []
bumper_labels = []
count = 0
c = 0
for key in test_annotation['test_im_names']:
	#print(key, " -> ", train_annotation[key])
	#print(key)
	#img_name = key
	#ID = int(img_name.split('_')[0])
	#type_label = train_annotation['type_label'][ID]
	color_label = test_annotation['color_label'][key]
	#type_label = test_annotation['type_label'][ID]
	#wheel_label = test_annotation['wheel_label'][ID]
	#luggage_label = test_annotation['luggage_label'][ID]
	#sky_label = test_annotation['sky_label'][ID]
	#bumper_label = test_annotation['bumper_label'][ID]
	print(color_label)
	break
	if color_label == 0:
		c+= 1
# 	if type_label==0:
# 		print(img_name)
# 		print(type_label)
# 		break
	color_labels.append(color_label)
	type_labels.append(type_label)
	wheel_labels.append(wheel_label)
	sky_labels.append(sky_label)
	luggage_labels.append(luggage_label)
	bumper_labels.append(bumper_label)
	#print(type_label)
	count+=1
# 	if count == 1:
# 		break
	
#ID = int(img_name.split('_')[0])

#wheel_label = train_annotation['wheel_label'][ID]
# type_label = train_annotation['type_label'][ID]
# color_label = train_annotation['color_label'][ID]
# luggage_label = train_annotation['luggage_label'][ID]
# sky_label = train_annotation['sky_label'][ID]
# bumper_label = train_annotation['bumper_label'][ID]

color_labels = np.array(color_labels)
type_labels = np.array(type_labels)
wheel_labels = np.array(wheel_labels)
sky_labels = np.array(sky_labels)
luggage_labels = np.array(luggage_labels)
bumper_labels = np.array(bumper_labels)

print(np.unique(type_labels))
#print(np.count_nonzero(color_labels == 0))
count_list = []
count_type = []
count_luggage = []
count_sky = []
count_bumper = []
count_wheel = []
for i in range(9):
	count_list.append(np.count_nonzero(color_labels==i))
	
for i in range(7):
	count_type.append(np.count_nonzero(type_labels==i))
	
for i in range(2):
	count_luggage.append(np.count_nonzero(luggage_labels==i))
	count_sky.append(np.count_nonzero(sky_labels==i))
	count_bumper.append(np.count_nonzero(bumper_labels==i))
	count_wheel.append(np.count_nonzero(wheel_labels==i))
print(np.count_nonzero(type_labels==0))
#print(color_labels)
print(count_type)
#image = io.imread(os.path.join(drive, directory, "/images_train/00000000_0001_00000001.jpg"))
#image = io.imread((drive+directory+"/images_train/00000000_0001_00000001.jpg"))
#train_annotation['']
x = ['White', 'Black', 'Gray', 'Red', 'Green', 'Blue', 'Yellow', 'Brown', 'Others']
x_type = ['Sedan', 'Hatchback', 'SUV', 'Bus', 'Lorry', 'Truck', 'Others']
x_binary = ['No', 'Yes']

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, constrained_layout=False, figsize=(15, 10))

#fig = plt.figure(constrained_layout=True)

ax1=plt.subplot(2, 2, 1)
ax1.bar(x_binary, count_luggage)
ax1.set_xlabel('Luggage rack?')
ax1.set_ylabel('# of Images') 
ax1.set_title('Frequency distribution of luggage rack in training set')

ax2 = plt.subplot(2,2,2)
ax2.bar(x_binary, count_bumper)
ax2.set_xlabel('Bumper?')
ax2.set_ylabel('# of Images') 
ax2.set_title('Frequency distribution of bumper in training set')

ax2 = plt.subplot(2,2,3)
ax2.bar(x_binary, count_wheel)
ax2.set_xlabel('Spare tire?')
ax2.set_ylabel('# of Images') 
ax2.set_title('Frequency distribution of spare tire in training set')


ax2 = plt.subplot(2,2,4)
ax2.bar(x_binary, count_sky)
ax2.set_xlabel('skylight?')
ax2.set_ylabel('# of Images') 
ax2.set_title('Frequency distribution of skylight in training set')


#plt.title('Frequency distribution of vehicle color and type in training set')
plt.show()
#face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    #root_dir='data/faces/')