import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load MINST data
(x, y), (x_test, y_test) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# data range of x, y
print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# divide batch
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch', sample[0].shape, sample[1].shape)

# fully connected layer
w1 = tf.Variable(tf.random.truncated_normal([28*28, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# 学习率
lr = 0.001

# iterate 60k data 10 times
for epoch in range(10):
    # record steps and print out info
    # loop through 60k
    # every batch [128]
    for step, (x, y) in enumerate(train_db):
        # x:[128, 28, 28]
        # y:[128]

        # 拍平x
        x = tf.reshape(x, [-1, 28*28])

        # for integration
        with tf.GradientTape() as tape:
            # h1 = x@w1+b1
            # [b,784]@[784, 256] + [256] -> [b, 256] + [256] -> [b, 256]
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

        # compute loss
        # [b, 10] - [b, 10]
        # [b]
        loss = tf.reduce_mean(tf.square(out - y))

        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # update weights
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))

