import torch
import torch.nn as nn
from blocks import ShuffleV1Block

#PyTorch 中定义一个神经网络模型的方式。具体来说，它表示一个继承自 nn.Module 类的自定义类，这个类定义了 ShuffleNetV1 模型。
class ShuffleNetV1(nn.Module):
    # def __init__(self, input_size=224, n_class=1000, model_size='2.0x', group=None): #代码表示，定义初始化self，输入大小，model_size 是 ShuffleNet 的一个关键超参数，用于控制模型的复杂度。
    def __init__(self, input_size=224, n_class=102, model_size='2.0x', group=3): #代码表示，定义初始化self，输入大小，model_size 是 ShuffleNet 的一个关键超参数，用于控制模型的复杂度。
        super(ShuffleNetV1, self).__init__()  #调用父类 nn.Module 的构造函数，初始化父类的属性。
        print('model size is ', model_size)

        #assert group is not None 是 Python 中的一种断言语句，用于检查变量 group 是否为 None。
        # 如果 group 为 None，则程序会抛出一个 AssertionError，并停止执行。简单来说，这个语句用于确保 group 在后续的代码中是有效的（即不是 None），从而避免因 None 值导致的潜在错误。
        assert group is not None
        #self.stage_repeats 是 ShuffleNet 等网络中的一个超参数，用来控制各个阶段（stage）中模块的重复次数。
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        #判断分组情况，分为多少组
        #self.stage_out_channels 用于存储 ShuffleNet 网络各个阶段的输出通道数。它是根据 model_size 来调整的，可以帮助控制网络的深度和复杂度。
        if group == 3:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 12, 120, 240, 480]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 240, 480, 960]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 360, 720, 1440]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 480, 960, 1920]
            else:
                raise NotImplementedError
        elif group == 8:
            if model_size == '0.5x':
                self.stage_out_channels = [-1, 16, 192, 384, 768]
            elif model_size == '1.0x':
                self.stage_out_channels = [-1, 24, 384, 768, 1536]
            elif model_size == '1.5x':
                self.stage_out_channels = [-1, 24, 576, 1152, 2304]
            elif model_size == '2.0x':
                self.stage_out_channels = [-1, 48, 768, 1536, 3072]
            else:
                raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        # nn.Sequential
        # 是一个容器，用来按顺序组合多个层（如卷积层、批量归一化层和激活函数）。它确保了各个层按顺序执行，
        # 每个层的输出将作为下一个层的输入。nn.Sequential
        # 可以让代码更加简洁，特别是在构建一系列顺序操作时。
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )
        #这段代码通过一个 3x3 的最大池化层，对输入特征图进行下采样，减少分辨率，
        # 同时保留显著特征。设置的参数（kernel_size=3, stride=2, padding=1）是深度学习模型中常用的配置，
        # 尤其适合在早期阶段对大尺寸输入图像进行快速降维的任务。
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # 用于构建整个网络的 特征提取层。其核心在于使用循环来创建多个阶段（stages），
        # 每个阶段包含若干个 ShuffleNetV1Block，并将这些模块存储为 self.features。
        self.features = []
        # 外层循环：阶段划分，shufflenet的结构，表示len（stage_repeats）=3，表示三个阶段
        for idxstage in range(len(self.stage_repeats)):
            #每个阶段的重复次数
            numrepeat = self.stage_repeats[idxstage]
            # 输出通道数
            output_channel = self.stage_out_channels[idxstage+2]
            #内层循环 numrepeat=4，8，4，表示重复次数
            for i in range(numrepeat):
                # 三元表达式，显示stride，表示当前阶段的第一个模块（i == 0），步幅设置为 2，用于下采样；否则，步幅为 1
                stride = 2 if i == 0 else 1
                # 如果是第一个阶段的第一个模块（即整个网络的第一个块），则设置 first_group=True，表示使用不同的组卷积配置。
                first_group = idxstage == 0 and i == 0
                # 这是 ShuffleNet 的基本单元模块，具体实现会包括分组卷积、通道混洗等操作。
                self.features.append(ShuffleV1Block(input_channel, output_channel,
                                            group=group, first_group=first_group,
                                            mid_channels=output_channel // 4, ksize=3, stride=stride))
                # 每次循环后，input_channel 被更新为当前模块的输出通道数（output_channel），这样下一次模块的输入通道数就能正确对接。
                input_channel = output_channel
        # self.features 是之前通过多个 ShuffleV1Block 堆叠而成的模块，用于提取输入图像的深层特征。
        # 它已经被包装成了一个 nn.Sequential，按顺序执行所有模块。
        self.features = nn.Sequential(*self.features)

        self.globalpool = nn.AvgPool2d(7)
        # 这是一个全连接层（Fully Connected Layer），用于将特征映射到分类任务的类别数。
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        # 这个方法用于初始化网络中的权重参数，确保网络在训练时以合理的初始值开始。
        self._initialize_weights()

    #前向传播（forward） 方法，用于描述输入数据如何依次经过模型的各个部分，直到得到最终的输出
    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)

        x = self.globalpool(x)
        x = x.contiguous().reshape(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x
    #定义了一个私有方法 _initialize_weights，用于初始化 ShuffleNet 模型中的权重和偏置。
    # 这一步非常重要，因为合理的权重初始化可以提高模型的训练效率并加速收敛。
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":  #main做测试，
    model = ShuffleNetV1(group=3)
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())