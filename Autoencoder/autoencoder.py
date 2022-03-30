import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt
from keras import Sequential, layers

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


# 多张图片拼接成一张
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


# [784] => [20]
h_dim = 20
batchsz = 512
lr = 1e-3

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# no label

train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, x_test.shape)


class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            # to [20]
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(28 * 28)
        ])

    def call(self, inputs, training=None):
        # [b, 784] => [b, 20]
        h = self.encoder(inputs)
        # [b, 20] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat


model = AE()
# 使用tuple类型
model.build(input_shape=(None, 784))
model.summary()

optimizer = tf.optimizers.Adam()

for epoch in range(100):

    for steps, x in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            # 重建后的x
            x_rec_logits = model(x)

            # 当作分类问题
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if steps % 100 == 0:
            print(epoch, steps, float(rec_loss))


        # evaluation
        x = next(iter(test_db))
        # flatten
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # 拼接
        # [b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x, x_hat], axis = 0)
        x_concat = x_hat
        # change to numpy image
        x_concat = x_concat.numpy() * 255
        x_concat = x_concat.astype(np.uint8)
        # 每epoch保存文件
        save_images(x_concat, 'ae_images/rec_epoch_%d.png'%epoch)

