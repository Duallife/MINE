import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60k, 28, 28]
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
#
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

# data range of x, y
print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# divide batch
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch', sample[0].shape, sample[1].shape)


# [b, 784] -> [b, 256] - [b,128] - [b, 10]
# [dim_in, dim_out], [dim_out]

# tape method only track tf.Variable (prevent error in gradient computation (NONE))

# stddev 避免梯度爆炸/消失
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
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
            # 256->128
            h2 = h1@w2 + b2
            h1 = tf.nn.relu(h2)
            # 128->10
            out = h2@w3 + b3

            # computer loss
            # out: [b, 10]
            # y: change one_hot
            # [b] -> [b, 10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)

        # gradient computation
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print(grads)
        # w1 = w1 - lr * w1 grad

        # keep as tf.variable
        w1.assign_sub(lr * grads[0])
        # change to tf.tensor while subtracting
        # w1 = w1 - lr * grads[0]

        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        # print(isinstance(b3, tf.Variable))
        # print(isinstance(b3, tf.Tensor))

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))




