import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

dropout_rate = tf.placeholder("float")

x = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 Bitmap image
y_ = tf.placeholder(tf.float32, [None, 10]) # answer (0~9)

W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable("W4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable("W5", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())

b1 = tf.Variable(tf.random_normal([256]))      # Bias for hidden layer 1
b2 = tf.Variable(tf.random_normal([256]))      # Bias for hideen layer 2
b3 = tf.Variable(tf.random_normal([256]))      # Bias for hideen layer 3
b4 = tf.Variable(tf.random_normal([256]))      # Bias for hideen layer 4
b5 = tf.Variable(tf.random_normal([10]))       # Bias for output


# Fully connected layer -> Softmax
_L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(_L1, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
L3 = tf.nn.dropout(_L1, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
L4 = tf.nn.dropout(_L1, dropout_rate)
hypothesis = tf.add(tf.matmul(L4, W5), b5) # model

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y_))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(y_,1)) # How may has matched
label = tf.argmax(hypothesis,1)
answer = tf.argmax(y_,1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(10001):
        batch_xs, batch_ys = mnist.train.next_batch(100) # Batch 100 images
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys, dropout_rate:0.7}) # Run training with batch
        if i % 100 == 0: # For each 100 steps, print out current accuracy
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels, dropout_rate:1}))
#            print(sess.run(label, feed_dict={x: mnist.test.images})[0:20])
#            print(sess.run(answer, feed_dict={ y_:mnist.test.labels})[0:20])

