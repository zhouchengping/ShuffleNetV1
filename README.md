ShuffleNetV1的复现

其中数据集是5种花卉的数据


训练集和测试集的格式为
![image](https://github.com/user-attachments/assets/1886f21b-d208-4cdc-a3cd-4e9a97613a02)

其中blocks.py为网络的基本网络，network.py为整体网络模型，train.py为训练的启动文件，simpletest.py为单张图片的测试。
