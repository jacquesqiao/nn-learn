import tensorflow as tf
import numpy as np

#device grpc://localhost:2500
# with tf.device("/job:worker/task:1"):
#     b = tf.constant("Hello, distributed TensorFlow!")

b = tf.constant("Hello, distributed TensorFlow!")

with tf.Session("grpc://10.87.171.23:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(b)


train_X = np.linspace(-1, 1, 101)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
X = tf.placeholder("float")
Y = tf.placeholder("float")

with tf.device("/job:worker/task:0"):
    w = tf.Variable(0.0, name = "weight")

with tf.device("/job:worker/task:1"):
    b = tf.Variable(0.0, name = "reminder")

init_op = tf.initialize_all_variables()
cost_op = tf.square(Y - tf.mul(X, w) - b)

with tf.device("/job:worker/task:0"):
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost_op)

with tf.Session("grpc://szwg-hadoop-l00278.szwg01.baidu.com:8201", config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init_op)
    for i in range(10):
      for (x, y) in zip(train_X, train_Y):
        sess.run(train_op, feed_dict={X:x, Y:y})
    print(sess.run(w))
    print(sess.run(b))