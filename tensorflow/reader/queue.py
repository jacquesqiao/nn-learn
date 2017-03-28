
import tensorflow as tf

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1, 0.2, 0.3],))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

print "+++"
print type(init)
print "+++"
print q.size()

with tf.Session() as sess:
    sess.run(init)
    quelen = sess.run(q.size())
    for i in range(2):
        sess.run(q_inc)

    quelen = sess.run(q.size())
    for i in range(quelen):
        print (sess.run(q.dequeue()))

