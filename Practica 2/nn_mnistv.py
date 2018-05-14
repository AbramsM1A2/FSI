import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import tensorboard


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

#
# plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print (train_y[57])


test_y_data = one_hot(test_y.astype(int), 10)
valid_y_data = one_hot(valid_y.astype(int), 10)

# TODO: the neural net!!
x_data = train_x
y_data = one_hot(train_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)
tf.summary.histogram("softmax_w2", W1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)
tf.summary.histogram("softmax_w2", W2)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))  # error

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20  # lote

training_errors = []
validation_errors = []
epoch = 0
validation_error = 0.1
difference = 100.0

while difference > 0.001:
    epoch += 1
    for jj in range(len(x_data) // batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print("------------Training error-----------")
    training_error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    training_errors.append(training_error)
    print("Epoch #:", epoch, "Training Error: ", training_error)

    print("------------Accuracy-----------")
    print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

    print("------------Validacion-----------")
    validation_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y_data})
    validation_errors.append(validation_error)
    if epoch > 1:
        difference = validation_errors[-2] - validation_error

    print("Epoch #:", epoch, "Validation Error: ", validation_error)

print("------------END TRAINING------------------")

print("----------------------")
print("   Start testing...  ")
print("----------------------")
error = 0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y_data, result):
    if np.argmax(b) != np.argmax(r):
        error += 1
    print(b, "-->", r)

success = 100 - (error * 100 / 10000)
error = error * 100 / 10000
print("--------------------------")
print("Fallos:", error, "%")
print("Aciertos:", success, "%")
print("----------END TESTING---------")

# plt.figure()
# plt.plot(aepoch, aerror)
# plt.plot(aepoch, avalid)
# plt.xlabel("Iteraciones", fontsize=20)
# plt.ylabel("Error", fontsize=20)
# plt.show()

plt.ylabel('Errors')
plt.xlabel('Epochs')
training_line, = plt.plot(training_errors)
plt.savefig('mnist.png')
