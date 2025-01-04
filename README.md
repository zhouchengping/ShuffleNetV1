ShuffleNetV1的复现
文件夹：selfData为自定义的五种花卉类型的shufflenet代码
Oxford Flowers 102文件夹为这个数据集下的代码

训练集和测试集的格式为
![image](https://github.com/user-attachments/assets/1886f21b-d208-4cdc-a3cd-4e9a97613a02)

其中blocks.py为网络的基本网络，network.py为整体网络模型，train.py为训练的启动文件，simpletest.py为单张图片的测试。
配置好环境
训练时可以使用pychar进行运行，可以直接使用python train.py


