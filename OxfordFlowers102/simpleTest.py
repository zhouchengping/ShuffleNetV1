import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import PIL
from PIL import Image
from network import ShuffleNetV1


# 假设你有一个类标签映射
class_names = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15','c16','c17','c18','c19','c20','c21','c22','c23','c24','c25','c26','c27','c28','c29','c30','c31','c32','c33','c34','c35','c36','c37','c38','c39','c40','c41','c42','c43','c44','c45','c46','c47','c48','c49','c50','c51','c52','c53','c54','c55','c56','c57','c58','c59','c60','c61','c62','c63','c64','c65','c66','c67','c68','c69','c70','c71','c72','c73','c74','c75','c76','c77','c78','c79','c80','c81','c82','c83','c84','c85','c86','c87','c88','c89','c90','c91','c92','c93','c94','c95','c96','c97','c98','c99','c100','c101','c102']

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:,::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:,::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:,::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


# 加载 ShuffleNetV1 模型
def load_model(model_path):
    # 创建模型实例
    print('创建模型实例')
    # 根据你的类别数量调整
    model = ShuffleNetV1(n_class=102,group=3)
    # 根据你的类别数量调整
    checkpoint = torch.load(model_path)  # 加载模型文件
    # 处理 `module.` 前缀问题
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k  # 去掉 'module.' 前缀
        new_state_dict[new_key] = v
        # print(new_key)
    model.load_state_dict(new_state_dict)  # 加载调整后的模型权重
    # model.load_state_dict(state_dict)  # 加载调整后的模型权重
    model.eval()  # 切换到评估模式
    return model


# 图片预处理
def preprocess_image(image_path):
    # 定义图片预处理
    print("调整图片大小,中心裁剪成224x224,ToBGRTensor")
    preprocess = transforms.Compose([
        # OpencvResize(256),  # 调整图片大小
        # transforms.RandomResizedCrop(224),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # ToBGRTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        OpencvResize(256),
        transforms.CenterCrop(224),
        ToBGRTensor(),
    ])
    #     transforms.Resize(256),  # 调整图片大小
    #     transforms.RandomResizedCrop(224),
    #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #     transforms.ToTensor(),  # 转换为tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    # ])


    # 打开图片并进行预处理
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # 增加一个批次维度
    # print(image)
    return image


# 预测类别
def predict(model, image):
    with torch.no_grad():  # 不需要梯度计算
        output = model(image)  # 获取模型的输出
        print('output:',output)
        probabilities = F.softmax(output, dim=1)  # softmax 计算概率
        predicted_class = torch.argmax(probabilities, dim=1).item()  # 获取最大概率对应的类别
    return predicted_class, probabilities


# 主函数
def main(image_path, model_path):
    print("加载模型")
    # 加载模型
    model = load_model(model_path)
    print("预处理图片")
    # 预处理图片
    image = preprocess_image(image_path)
    print("进行预测")
    # 进行预测
    predicted_class, probabilities = predict(model, image)

    # 打印预测结果
    print(f'Predicted Class: {class_names[predicted_class]}')
    print(f'Prediction Probabilities: {probabilities}')

    # 显示图片和预测结果
    image_display = Image.open(image_path)
    plt.imshow(image_display)
    plt.title(f"Predicted Class: {class_names[predicted_class]}")
    plt.show()


if __name__ == '__main__':
    image_path = 'data/val/c25/image_06512.jpg'  # 替换为你要验证的图片路径
    model_path = 'models/checkpoint-010000.pth.tar'  # 替换为训练后的 .pth.tar 文件路径
    main(image_path, model_path)
