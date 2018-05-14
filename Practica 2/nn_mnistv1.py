import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='bytes')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print (train_y[57])


# TODO: the neural net!!
sess = tf.InteractiveSession()

x_data = train_x  # the data
y_data = one_hot(train_y, 10)  # the labels

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(16)) * 0.1)

#h = tf.nn.sigmoid(tf.matmul(x, W) + b)
h = tf.matmul(x, W) + b  # Try this!
y = tf.matmul((h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))  # error

tf.scalar_summary("loss", loss)

sess.run(tf.global_variables_initializer())

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# tf.global_variables_initializer().run()


print("----------------------")
print("   Start training...  ")
print("----------------------")


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


batch_size = 20  # lote

for epoch in range(100):
    for jj in range(len(x_data) // batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))

muestra_x = test_x[0:20]
muestra_y = test_y[0:20]

result = sess.run(y, feed_dict={x: muestra_x})
for b, r in zip(muestra_y, result):
    print(b, "-->", r)
    print("----------------------------------------------------------------------------------")

# est_x_muestra = test_x[0:20]
# test_y_muestra = test_y[0:20]
# result = sess.run(y, feed_dict={x: test_x_muestra})
# for b, r in zip(test_y_muestra, result):
#   print (b, "-->", r)
# print("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))

# for epoch in range(1000):
#   batch_xs,batch_ys = next_batch(100,train_x,train_y)
#   #train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
#   sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
#   print("Epoch #:", epoch, "Error: ", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
#   result = sess.run(y, feed_dict={x: batch_xs})
#   for b, r in zip(batch_ys, result):
#       print(b, "-->", r)
#   print("----------------------------------------------------------------------------------")
