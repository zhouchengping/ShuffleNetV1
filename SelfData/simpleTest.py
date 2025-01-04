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
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

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
    model = ShuffleNetV1(n_class=5,group=3)  # 根据你的类别数量调整
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
    image_path = 'data/train/tulips/132538272_63658146d9_n.jpg'  # 替换为你要验证的图片路径
    model_path = 'models/checkpoint-020000.pth.tar'  # 替换为训练后的 .pth.tar 文件路径
    main(image_path, model_path)
