import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
from Resnet import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)

    return x, y


# load data
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print('Init shape: ', x.shape, y.shape, x_test.shape, y_test.shape)

# db batch
batchsize = 128
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(10000).map(preprocess).batch(batchsize)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsize)

# two dims
db_iter = next(iter(train_db))
print('shape : ', db_iter[0].shape, db_iter[1].shape, tf.reduce_min(db_iter[0]), tf.reduce_max(db_iter[0]))


def main():
    model = resnet18(100)
    model.build(input_shape=(None, 32, 32, 3))
    # summary ////////////////////////////////////////////////////////////////////
    model.summary()

    optimizer = optimizers.Adam(learning_rate=0.001)

    for epoch in range(50):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model(x)

                y_onehot = tf.one_hot(y, depth=100)
                # loss computation
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0

        for x, y in test_db:
            logits = model(x)
            # add up to 1
            prob = tf.nn.softmax(logits, axis=1)
            # pick out the highest
            pred = tf.argmax(prob, axis=1, output_type=tf.int32)
            # pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'acc= ', acc)


if __name__ == '__main__':
    main()
