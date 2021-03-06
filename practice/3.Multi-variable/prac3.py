import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data

W = tf.Variable(tf.random_uniform([1,len(x_data)], -5.0, 5.0))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
epoch = []
for step in xrange(2001):
    sess.run(train)
    epoch.append(sess.run(cost))
    if step % 20 == 0:
        print step, sess.run(cost), sess.run(W)

plt.plot(epoch[1:1000])
plt.show()
