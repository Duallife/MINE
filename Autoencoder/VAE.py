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

z_dim = 10


class VAE(keras.Model):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        # layer before Mean & Variance layer
        self.fc1 = layers.Dense(128)
        # mean predict
        self.fc2 = layers.Dense(z_dim)
        # variance predict
        self.fc3 = layers.Dense(z_dim)

        # Decoder
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(28 * 28)

    def encoder(self, x):
        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)
        # get variance
        # more stable (change 0 -> positive inf to all Real number)
        log_var = self.fc3(h)

        return mu, log_var

    def decoder(self, z):
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    # reparameterization trick (CORE!)
    def reparameterize(self, mu, log_var):
        # random node
        eps = tf.random.normal(log_var.shape)
        # remove log -> square root it
        std = tf.exp(log_var) ** 0.5
        # add them together
        z = mu + std * eps
        return z

    def call(self, inputs, training=None):
        # [b, 784] => []
        mu, log_var = self.encoder(inputs)
        # reparameterization trick
        z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)
        # use mu & log_var to computer kl divergence
        return x_hat, mu, log_var


model = VAE()
# can't run random for shape None
model.build(input_shape=(4, 784))
optimizer = tf.optimizers.Adam(lr)

for epoch in range(1000):

    for step, x in enumerate(train_db):

        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)

            # same as autoencoder
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

            # compute kl divergence (mu, var) ~ N (0, 1)
            # q normal distribution (sigma2 = 1, mu2 = 0) sigma = std = sqrt(var)
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_mean(kl_div)

            loss = rec_loss + 1. * kl_div

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step * 100 == 0:
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss', float(rec_loss))

    # evaluation

    # decoder input
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    # change to image
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/sampled_epoch%d.png' %epoch)

    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_rec_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/rec_epoch%d.png' %epoch)
