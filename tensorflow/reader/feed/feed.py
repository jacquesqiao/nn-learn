import tensorflow as tf

"""
fully_connected_feed.py
"""
a1 = tf.placeholder(tf.int16)
a2 = tf.placeholder(tf.int16)
b = tf.add(a1, a2)

li1 = [2, 3, 4]
li2 = [4, 0, 1]

with tf.Session() as sess:
    print sess.run(b, feed_dict={a1: li1, a2: li2})
