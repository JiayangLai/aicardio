# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from net_lib.classifation_end import CLS_end
from net_lib.model_builder import model_builder_classification
from utils.check_mkdir import chk_mk_model
from utils.saved_path_gen import savedpathgen


def train_and_valid(model, loss_function, optimizer, net_in, num_begin,epochs, train_data_size, valid_data_size):
    device = torch.device("cuda:0")
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("\nEpoch: {}/{}".format(num_begin+epoch + 1, num_begin+epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in tqdm(enumerate(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()

            outputs = model(inputs)
            # print('outputs',outputs)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            # print('\nlabels     ', labels)
            # print('predictions', predictions)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()

            for j, (inputs, labels) in tqdm(enumerate(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                # print(outputs)

                loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, dim=1)
                # print('\nlabels     ', labels)
                # print('predictions', predictions)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = num_begin+epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.5f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.5f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                num_begin+epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.5f} at epoch {:03d}".format(best_acc, best_epoch))

        torch.save(model, 'models_saved/' + net_in + '/TL{:.7f}_TA{:.3f}_VL{:.5f}_VA{:.1f}_EP'.format(
            avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100
        ) + str(num_begin+epoch + 1) + '.pt')
    return model, history


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_num_threads(1)  # 减少cpu的占用率


image_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        # transforms.RandomRotation(degrees=15),
        transforms.Resize(size=(256)),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                       [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(256)),
        transforms.CenterCrop(size=256),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                       [0.229, 0.224, 0.225])
    ])
}

batch_size = 32
num_classes = 5

data_path = 'data_cardio'

dataset = {
    'train': datasets.ImageFolder(root=data_path + '/train', transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=data_path + '/test', transform=image_transforms['valid'])
}

print('train class_to_idx:', dataset['train'].class_to_idx)
print('valid class_to_idx:', dataset['valid'].class_to_idx)

trn_size = len(dataset['train'])
val_size = len(dataset['valid'])

train_data = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=True)

net_name = 'resnext50_32x4d'
pretrained = True
isretrain = True
retrain_mod_name = 'TL0.0010272_TA99.972_VL0.03416_VA98.6_EP9'
model_saved_path = savedpathgen(pretrained, net_name)


if isretrain:
    numbegin = int(retrain_mod_name.split('EP')[1])
    model_now = torch.load('./models_saved/' + model_saved_path + '/' + retrain_mod_name + '.pt')
else:
    model_now = model_builder_classification(net_name, num_classes, pretrained)


model_now = model_now.to('cuda:0')

loss_func = nn.NLLLoss()
optimizer = optim.Adam(model_now.parameters())

num_epochs = 50

chk_mk_model(model_saved_path)
trained_model, history = train_and_valid(model_now, loss_func, optimizer, model_saved_path, numbegin, num_epochs, trn_size,
                                         val_size)
torch.save(history, './models_saved/' + model_saved_path + '/history.pt')

# history = np.array(history)
# plt.plot(history[:, 0:2])
# plt.legend(['Tr Loss', 'Val Loss'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.ylim(0, 1)
# plt.savefig('_loss_curve.png')
# plt.show()
#
# plt.plot(history[:, 2:4])
# plt.legend(['Tr Accuracy', 'Val Accuracy'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)
# plt.savefig('_accuracy_curve.png')
# plt.show()
