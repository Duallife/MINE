import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import optimizers, preprocessing
import numpy as np
import os
from resnet import resnet18
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2222)
np.random.seed(2222)

PATH = './checkpoint/10.ckpt'

# load imagenet
(raw_train, raw_validation), metadata = tfds.load('imagenette/160px-v2',
                                                  split=['train', 'validation'],
                                                  shuffle_files=True,
                                                  batch_size=16,
                                                  with_info=True,
                                                  as_supervised=True
                                                  )

IMG_SIZE = 160
NUM_CLASSES = 10


# preprocess
def preprocessing(image, label):
    #   remove_black_borders
    # image = remove_black_borders(image[-3:])

    # image = tf.image.random_contrast(image, 0, 1)
    # image = tf.image.random_brightness(image, 0.5)
    image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    label = tf.cast(label, tf.int32)
    return image, label


validation = raw_validation.map(preprocessing)


def main():
    model = resnet18(NUM_CLASSES)
    model.load_weights(PATH)
    model.summary()

    total_correct = 0
    total_num = 0

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


if __name__ == '__main__':
    main()
