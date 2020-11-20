# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:58:55 2020

@author: udits
"""
import numpy as np
import matplotlib.pyplot as plt 

train = np.load('C:/MS/RA_Tracking/Metrics_28Sept/TypeTimes15_dropout_0.2/train_metrics_20e_dropout_02_type_15.npy', allow_pickle=True)
val = np.load('C:/MS/RA_Tracking/Metrics_28Sept/TypeTimes15_dropout_0.2/val_metrics_20e_dropout_02_type_15.npy', allow_pickle=True)

epochs = train[0]
acc_col_train = train[1]
acc_type_train = train[2]
loss_train = train[3]

print(val[9])
print(acc_type_train)
# epochs = val[:,0]
# acc_col_train = val[:,1]
# acc_type_train = val[:,2]
# loss_train = val[:,3]

# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.plot(epochs, acc_col_train)
# ax1.axis([0,20,0.5,1])
# ax1.set_title('Plot for Accuracy of color label vs. Epochs', fontsize=20)
# ax1.set_xlabel('Epochs', fontsize=20)
# ax1.set_ylabel('Accuracy for color label', fontsize=20)
# #ax1.set_xticks(fontsize=20)
# #ax1.set_yticks(fontsize=20)

# ax2.plot(epochs, acc_type_train)
# ax2.axis([0,20,0.5,1])
# ax2.set_title('PLot for Accuracy of type label vs. Epochs', fontsize=20)
# ax2.set_xlabel('Epochs', fontsize=20)
# ax2.set_ylabel('Accuracy for type label', fontsize=20)
# #ax2.set_xticks(fontsize=20)
# #ax2.set_yticks(fontsize=20)

# ax3.plot(epochs, loss_train)
# #ax3.axis([0,20,0,1])
# ax3.set_title('Change in loss with epochs', fontsize=20)
# ax3.set_xlabel('Epochs', fontsize=20)
# ax3.set_ylabel('Loss', fontsize=20)
#ax3.set_xticks(fontsize=20)
#ax3.set_yticks(fontsize=20)

# f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.plot(epochs, acc_col_train)
# ax1.axis([0,20,0.5,1])
# ax1.set_title('Plot for Accuracy of color label vs. Epochs (Validation set)', fontsize=20)
# ax1.set_xlabel('Epochs', fontsize=20)
# ax1.set_ylabel('Accuracy for color label', fontsize=20)
# #ax1.set_xticks(fontsize=20)
# #ax1.set_yticks(fontsize=20)

# ax2.plot(epochs, acc_type_train)
# ax2.axis([0,20,0.5,1])
# ax2.set_title('PLot for Accuracy of type label vs. Epochs (Validation set)', fontsize=20)
# ax2.set_xlabel('Epochs', fontsize=20)
# ax2.set_ylabel('Accuracy for type label', fontsize=20)
# #ax2.set_xticks(fontsize=20)
# #ax2.set_yticks(fontsize=20)

# ax3.plot(epochs, loss_train)
# #ax3.axis([0,20,0,1])
# ax3.set_title('Change in loss with epochs (Validation set)', fontsize=20)
# ax3.set_xlabel('Epochs', fontsize=20)
# ax3.set_ylabel('Loss', fontsize=20)
# #ax3.set_xticks(fontsize=20)
# #ax3.set_yticks(fontsize=20)

plt.show()
