import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # z: [b, 100] => [b, 3*3*512] => [b, 3, 3, 512] => [b, 64, 64, 3]
        # 升维
        self.fc1 = layers.Dense(3 * 3 * 512)
        # channel 大到小
        # feature size 小到大

        # Reverse conv
        self.conv1 = layers.Conv2DTranspose(256, kernel_size=3, strides=3, padding='valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        # 最后等于 64*64*3 for discriminator
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs, training=None):
        # [z, 100] => [z, 3*3*512]
        x = self.fc1(inputs)
        # [b, 3*3*512] => [b, 3, 3, 512]
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        #
        x = self.bn1(self.conv1(x), training=training)
        x = tf.nn.leaky_relu(x)
        x = self.bn2(self.conv2(x), training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)
        # 使用tanh,较为稳定
        # range [-1 to 1]
        x = tf.tanh(x)

        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        # [b, 64, 64, 3] => [b, 1] 二分问题
        # 分类器
        self.conv1 = layers.Conv2D(64, kernel_size=5, strides=3, padding='valid')
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w, c] => [b, -1] for 全连接
        self.flatten = layers.Flatten()
        # final output layer
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        # leaky_relu for stability
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))

        # [b, h, w, c] => [b, -1] for 全连接
        x = self.flatten(x)
        # [b, -1] => [b, 1]
        logits = self.fc(x)

        return logits


def main():
    d = Discriminator()
    g = Generator()

    x = tf.random.normal([2, 64, 64, 3])
    # hidden variable (input random scalar)
    z = tf.random.normal([2, 100])

    # 检查输出维度
    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)


if __name__ == '__main__':
    main()
