import glob
import os.path as osp
import random
from typing import Any
import numpy as np
import json
from PIL import Image

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, (0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


"""
image_file_path = "./1_image_classification/data/goldenretriever-3724972_640.jpg"
img = Image.open(image_file_path)

plt.imshow(img)
plt.show()



transform = ImageTransform(size, mean, std)
img_transformed = transform(img, phase="train")

img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)
plt.imshow(img_transformed)
plt.show()
"""
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def make_datapath_list(phase="train"):
    rootpath = "./1_image_classification/data/hymenoptera_data/"
    target_path = osp.join(rootpath + phase + "/**/*.jpg")
    # print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")


class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[53:57]
        elif self.phase == "val":
            label = img_path[51:55]

        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label


train_dataset = HymenopteraDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase="train"
)

val_dataset = HymenopteraDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase="val"
)

# index = 200
# print(train_dataset.__getitem__(index)[0].size())
# print(train_dataset.__getitem__(index)[1])

# 데이터 로더 작성하기
batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# batch_iterator = iter(dataloaders_dict["train"])
# inputs, labels = next(batch_iterator)

# print(inputs.size())
# print(labels)

# 네트워크 모델 작성하기 - 출력부 수정
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

net.train()

print("네트워크 설정 완료: 학습된 가중치를 읽어들여 훈련 모드로 설정했습니다.")

# 손실 함수 설정
criterion = nn.CrossEntropyLoss()

# 최적화 기법 설정
params_to_update = []

update_param_names = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

# print("-" * 10)
# print(params_to_update)

optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)


# 학습 및 검증 실시
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


num_epochs = 4
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
