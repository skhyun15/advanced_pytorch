import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

# %matplotlib inline

import torch
import torchvision
from torchvision import models, transforms

use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)
net.eval()


class BaseTransform:
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, img):
        return self.base_transform(img)


image_file_path = "./1_image_classification/data/goldenretriever-3724972_640.jpg"

img = Image.open(image_file_path)

# plt.imshow(img)
# plt.show()

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = np.clip(img_transformed, 0, 1)

# 가공 후 이미지의 모습
# plt.imshow(img_transformed)
# plt.show()


class ILSVRCPredictor:
    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name


ILSVTC_class_index = json.load(
    open("./1_image_classification/data/imagenet_class_index.json", "r")
)

predictor = ILSVRCPredictor(ILSVTC_class_index)

image_file_path = "./1_image_classification/data/goldenretriever-3724972_640.jpg"
img = Image.open(image_file_path)

transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)
inputs = img_transformed.unsqueeze_(0)

out = net(inputs)

result = predictor.predict_max(out)

print("예측 결과:", result)
