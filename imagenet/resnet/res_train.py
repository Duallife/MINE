import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import optimizers, preprocessing, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import numpy as np
import os
from resnet import resnet18
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2222)
np.random.seed(2222)

# load imagenet
(raw_train, raw_validation), metadata = tfds.load('imagenette/320px-v2',
                                                  split=['train', 'validation'],
                                                  shuffle_files=True,
                                                  batch_size=16,
                                                  with_info=True,
                                                  as_supervised=True
                                                  )

SHUFFLE_BUFFER_SIZE = 1000
raw_train = raw_train.shuffle(SHUFFLE_BUFFER_SIZE)

IMG_SIZE = 320
NUM_CLASSES = 10


# attempt to remove border
def remove_black_borders(image):
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]


# print image data 1
for image_batch, label_batch in raw_train.take(1):
    print(image_batch.shape)
    print(label_batch.shape)
    print('max: ', tf.reduce_max(image_batch[-1]), 'min: ', tf.reduce_min(image_batch[-1]))
    print('max: ', tf.reduce_max(label_batch), 'min: ', tf.reduce_min(label_batch))
    img = remove_black_borders(image_batch[1])
    # plt.imshow(img)
    # plt.show()


# preprocess
def preprocessing(image, label):
    #   remove_black_borders
    # image = remove_black_borders(image[-3:])

    # image = tf.image.random_contrast(image, 0, 1)
    # image = tf.image.random_brightness(image, 0.5)

    # image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    image = tf.cast(image, dtype=tf.float32) / 255.

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    label = tf.cast(label, tf.int32)
    return image, label


def aug(image, label):
    image = tf.image.random_contrast(image, 0, 1)
    image = tf.image.random_brightness(image, 0.5)
    image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label


train = raw_train.map(preprocessing)
# train1 = raw_train.map(aug)
# train = train.concatenate(train1)
validation = raw_validation.map(preprocessing)
# test = raw_test.map(preprocessing)
print(train)

# print image data 2
for image_batch, label_batch in train.take(1):
    print(image_batch.shape)
    print(label_batch.shape)
    print('max: ', tf.reduce_max(image_batch[-1]), 'min: ', tf.reduce_min(image_batch[-1]))
    # plt.imshow((image_batch[1]), vmin=-1, vmax=1)
    # plt.show()


def main():
    model = resnet18(NUM_CLASSES)
    model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    # optimizer1 = optimizers.Adam(learning_rate=0.01)
    optimizer2 = optimizers.Nadam(learning_rate=0.001)

    # visualize
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # load data
    # model.load_weights('./checkpoint/10.ckpt')
    INIT = 0
    EPOCH = 100

    for epoch in range(INIT, EPOCH):

        data = train.shuffle(SHUFFLE_BUFFER_SIZE)
        for step, (x, y) in enumerate(data):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 10]
                logits = model(x, training=True)

                y_onehot = tf.one_hot(y, depth=NUM_CLASSES)
                # loss computation
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)

            # if epoch > 2:
            #     optimizer1.apply_gradients(zip(grads, model.trainable_variables))
            # else:
            #     optimizer2.apply_gradients(zip(grads, model.trainable_variables))
            optimizer2.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss), step=epoch * 600 + step)

        total_num = 0
        total_correct = 0

        for x, y in validation:
            logits = model(x, training=False)
            # add up to 1
            prob = tf.nn.softmax(logits, axis=1)
            # pick out the highest
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'acc= ', acc)
        val_images = x[:2]
        val_images = tf.reshape(val_images, [-1, IMG_SIZE, IMG_SIZE, 3])
        # val_images = (val_images + 1) * 127.5
        with summary_writer.as_default():
            tf.summary.scalar('acc', float(acc), step=epoch * 600 + step)
            tf.summary.image("val", val_images, max_outputs=25, step=epoch * 600 + step)

        model.save_weights('./checkpoint/%d.ckpt' % epoch)
        if epoch % 5 == 0:
            model.save_weights('./checkpoint/%d.ckpt' % epoch)


if __name__ == '__main__':
    main()
