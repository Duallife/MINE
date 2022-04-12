import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers


class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # depends on input strides = stride
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # to same shape: stride = 1
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential([layers.Conv2D(filter_num, (1, 1), strides=stride)])
            # self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            # return original
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        # [b,h,w,c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # depends on stride
        identity = self.downsample(inputs)

        # add up two layers (resnet)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):

    # layer_dims = how many blocks in one block
    def __init__(self, layer_dims, num_classes=100):  # [2, 2, 2, 2] layer number
        super(ResNet, self).__init__()

        # stride can be 1 dim or 2 dims or more
        # start layer
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=1),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                # maintain expected size (2 then size/2)
                                layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
                                ])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        # stride = 2, reduce dimension
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output: [b, 512, h, w] unknown h & w !!!!!!!!!!!!!!!!!!!!!!
        # transform into [b, 512, 1, 1] => [b, 512]
        self.avgpool = layers.GlobalAvgPool2D()

        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        # start layer
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b,c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample (if input stride not = 1)
        # not fix dimension
        res_blocks.add(BasicBlock(filter_num, stride))

        # from 1-2 add one block, total two blocks for standard unit
        # blocks-1
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18(num_classes):
    # 4 * 2 (res_block) *2 (basic_block) + prev + end = 18
    return ResNet([2, 2, 2, 2], num_classes)


def resnet34(num_classes):
    return ResNet([3, 4, 6, 3], num_classes)
