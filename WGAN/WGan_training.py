import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import image
import glob
from WGAN import Generator, Discriminator

from dataset import make_anime_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 拼合图片
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    image.imsave(image_path, final_image)


def celoss_ones(logits):
    # logits [b, 1]
    # [b] = [1, 1, 1, 1,...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))

    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # logits [b, 1]
    # [b] = [0, 0, 0, 0,...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))

    return tf.reduce_mean(loss)


# WGAN--------------------------------------------------------------
def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]
    # [b, h, w, c]
    # for broading cast fill 1
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # [batchsz, 1 ,1 ,1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits, interplate)

    # grads:[b, h, w, c]
    # 打平
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)

    # final loss calculation
    # gradient penalty added for WGAN !!!
    # lamda = 1
    gp = gradient_penalty(discriminator, batch_x, fake_image)

    loss = d_loss_fake + d_loss_real + 1. * gp

    return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    # make loss close to 1
    loss = celoss_ones(d_fake_logits)

    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    assert tf.__version__.startswith('2.')

    # hyper parameters
    # parameter of hidden input
    z_dim = 100
    epochs = 3000000
    batchsz = 512
    learning_rate = 0.002
    is_training = True

    # add 'r' in front to remove error
    # add jpg to load vertain file type
    img_path = glob.glob(r'F:\Database\anime\*.jpg')

    dataset, img_shape, _ = make_anime_dataset(img_path, batchsz)
    print(dataset, img_shape)
    # print first sample
    sample = next(iter(dataset))
    print(sample, tf.reduce_min(sample).numpy(), tf.reduce_max(sample).numpy())

    # 迭代器无限循环
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    # beta_1 is common in GAN
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        # random sampling
        batch_z = tf.random.uniform([batchsz, z_dim], minval=-1, maxval=1)
        batch_x = next(db_iter)

        # train D
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # train G
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # print data
        if epoch % 50 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss', float(g_loss), 'gp', float(gp))

            z = tf.random.uniform([25, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'wgan-%d.png' % epoch)
            save_result(fake_image.numpy(), 5, img_path, color_mode='P')


if __name__ == '__main__':
    main()
