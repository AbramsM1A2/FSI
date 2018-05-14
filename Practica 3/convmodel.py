# -*- coding: utf-8 -*-

# Sample code to use string producer.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    # if type(x) == list:
    #    x = np.array(x)
    # x = x.flatten()
    # o_h = np.zeros((len(x), n))
    # o_h[np.arange(len(x)), x] = 1
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)  # [float(i)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):

    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=40, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["mydata/2/*.jpg", "mydata/3/*.jpg", "mydata/4/*.jpg"],
                                                    batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(
    ["mydata/valid_2/*.jpg", "mydata/valid_3/*.jpg", "mydata/valid_4/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["mydata/test_2/*.jpg", "mydata/test_3/*.jpg", "mydata/test_4/*.jpg"],
                                                  batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

y = tf.placeholder(tf.float32, [None, 3])
y_ = tf.placeholder(tf.float32, [None, 3])
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    training_errors = []
    validation_errors = []
    epoch = 0
    validation_error = 0.1
    difference = 100.0

    while difference > 0.001:
        epoch += 1
        sess.run(optimizer)
        # if epoch % 20 == 0:
        print("-----------------------", "Iter:", epoch, "----------------------")
        print("--------------Valid-----------------")
        print(sess.run(label_batch_valid))
        print(sess.run(example_batch_valid_predicted))
        print("Error:", sess.run(cost_valid))
        validation_error = sess.run(cost_valid)
        validation_errors.append(validation_error)

        print("------------Accuracy-----------")
        print(sess.run(accuracy, feed_dict={y: example_batch_train_predicted.eval(), y_: label_batch_train.eval()}))

        print("--------------Train-----------------")
        print(sess.run(label_batch_train))
        print(sess.run(example_batch_train_predicted))
        print("Error:", sess.run(cost))
        training_error = sess.run(cost)
        training_errors.append(training_error)

        if epoch > 1:
            difference = np.absolute(validation_errors[-2] - validation_error)

    print("------------END TRAINING------------------")

    print("----------------------")
    print("   Start testing...  ")
    print("----------------------")

    expected_result = sess.run(label_batch_test)
    result = sess.run(example_batch_test_predicted)

    hits = 0
    mistakes = 0

    for b, r in zip(expected_result, result):
        if (np.argmax(b) == np.argmax(r)):
            hits += 1
        else:
            mistakes += 1
        print(b, "-->", r)
        print("Aciertos: ", hits)
        print("Fallos: ", mistakes)
        Total = hits + mistakes
        print("Porcentaje de aciertos: ", (float(hits) / float(Total)) * 100, "%")
        print("----------------------------------------------------------------------------------")

    plt.ylabel('Error')
    plt.xlabel('Epoca')
    training_line, = plt.plot(training_errors)
    plt.savefig('convolutiva.png')

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
