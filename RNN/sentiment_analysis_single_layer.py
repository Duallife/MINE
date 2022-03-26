# simple RNN cell

import os
import keras_preprocessing.sequence
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets, optimizers

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# the most frequent words (limit number)
total_words = 10000
max_review_len = 80
# how many num present one word
embedding_len = 100
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
# [b, 80], [b,80]
# every sentence is 80 after padding
x_train = keras_preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras_preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

batchsz = 128
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)  # drop remaining if cant be integer divide
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
# y = 1 = nice, y = 0 = bad
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class MyRNN(keras.Model):

    # units = h_dim
    def __init__(self, units):
        super(MyRNN, self).__init__()

        # all memory initialize as all zero
        # [b, 64]
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        # layer 1
        # transform text to embedding representation
        # [b, 80] => [b, 80, embedding_len = 100]
        # input_length = length of sentence = 80
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        # layer 2
        # [b, 80, 100], h_dim: 64
        # => [b, 64]
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.2)

        # layer 3
        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x, training = True) : train mode (with dropout)
        net(x, training = False)
        :param inputs: [b, 80]
        :param training: training state or not
        :return: [b, 1]
        """

        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)

        # embedding
        # [b, 80, 100] => [b, 64]
        # axis = 1 展开
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            # h1 = x*wxh+h0*whh
            # refresh state
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


def main():
    units = 64
    epochs = 4

    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)


if __name__ == '__main__':
    main()
