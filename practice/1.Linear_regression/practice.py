import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#hypothesis
hypothesis = W * x_data + b +2

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(1)
optimizer= tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# print "after W:" + sess.run(W) + " b:" + sess.run(b)
# Before starting, initailize the variables. We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)



print sess.run(hypothesis)
for step in xrange(2001):
        prev = sess.run(cost)
        sess.run(train)
        if prev-sess.run(cost) < 0.00001:
                break
        if step % 20 == 0:
                print step, sess.run(cost), sess.run(W), sess.run(b)


