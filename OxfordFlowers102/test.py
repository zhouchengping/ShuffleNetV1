import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from network import ShuffleNetV1  # 需要你自己提供ShuffleNetV1的模型定义文件

# model_path = 'models/checkpoint-005000.pth.tar'  # 替换为训练后的 .pth.tar 文件路径
# checkpoint = torch.load(model_path)
# print(checkpoint['state_dict'].keys())  # 打印出模型中包含的所有键
# # 打印文件内容，查看其键（key）
for i in range(103):
    print('\'c%d\''%i,end=',')

# print(checkpoint.keys())
# print(checkpoint['state_dict'].keys())
# print('/n')
# print(checkpoint['state_dict']['module.first_conv.1.weight'])