import pylayer as L
from sequential import Sequential

class VGG16(Sequential):
    def __init__(self, input_channel=3, output_class=10):
        super().__init__([
            # 第一层卷积层
            # 输入3通道图像，输出64通道特征图，卷积核大小3x3，步长1，填充1
            L.Conv2d(input_channel, output_channel=64, kernel_size=3, stride=1, padding=1),
            # 对64通道特征图进行Batch Normalization
            #L.BatchNorm2d(64),
            # 对64通道特征图进行ReLU激活函数
            L.ReLU(),
            # 输入64通道特征图，输出64通道特征图，卷积核大小3x3，步长1，填充1
            L.Conv2d(input_channel=64, output_channel=64, kernel_size=3, stride=1, padding=1),
            # 对64通道特征图进行Batch Normalization
            #L.BatchNorm2d(64),
            # 对64通道特征图进行ReLU激活函数
            L.ReLU(),
            # 进行2x2的最大池化操作，步长为2
            L.MaxPool2d(kernel_size=2, stride=2),

            # 第二层卷积层
            # 输入64通道特征图，输出128通道特征图，卷积核大小3x3，步长1，填充1
            L.Conv2d(input_channel=64, output_channel=128, kernel_size=3, stride=1, padding=1),
            # 对128通道特征图进行Batch Normalization
            #L.BatchNorm2d(128),
            # 对128通道特征图进行ReLU激活函数
            L.ReLU(),
            # 输入128通道特征图，输出128通道特征图，卷积核大小3x3，步长1，填充1
            L.Conv2d(input_channel=128, output_channel=128, kernel_size=3, stride=1, padding=1),
            # 对128通道特征图进行Batch Normalization
            #L.BatchNorm2d(128),
            L.ReLU(),
            # 进行2x2的最大池化操作，步长为2
            L.MaxPool2d(2, 2),

            # 第三层卷积层
            # 输入为128通道，输出为256通道，卷积核大小为33，步长为1，填充大小为1
            L.Conv2d(input_channel=128, output_channel=256, kernel_size=3, stride=1, padding=1),
            # 批归一化
            #L.BatchNorm2d(256),
            L.ReLU(),

            L.Conv2d(input_channel=256, output_channel=256, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(256),
            L.ReLU(),

            L.Conv2d(input_channel=256, output_channel=256, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(256),
            L.ReLU(),

            L.MaxPool2d(2, 2),

            # 第四层卷积层
            L.Conv2d(input_channel=256, output_channel=512, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(512),
            L.ReLU(),

            L.Conv2d(input_channel=512, output_channel=512, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(512),
            L.ReLU(),

            L.Conv2d(input_channel=512, output_channel=512, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(512),
            L.ReLU(),

            L.MaxPool2d(2, 2),

            # 第五层卷积层
            L.Conv2d(input_channel=512, output_channel=512, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(512),
            L.ReLU(),

            L.Conv2d(input_channel=512, output_channel=512, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(512),
            L.ReLU(),

            L.Conv2d(input_channel=512, output_channel=512, kernel_size=3, stride=1, padding=1),
            #L.BatchNorm2d(512),
            L.ReLU(),

            L.MaxPool2d(2, 2),

            L.Flatten(),
            L.Linear(512, 512),
            L.ReLU(),
            L.Linear(512, 256),
            L.ReLU(),
            L.Linear(256, output_class),
            
            L.CrossEntropyLossWithSoftmax(),
        ])

      
