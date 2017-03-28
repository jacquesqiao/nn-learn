import tensorflow as tf

# filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
filename_queue = tf.train.string_input_producer(['file0.csv', 'file1.csv'])
reader = tf.TextLineReader()

key, value = reader.read(filename_queue)


# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
# features = tf.concat(0, [col1, col2, col3, col4])
features = [col1, col2, col3, col4]

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label, res_key = sess.run([features, col5, key])

  coord.request_stop()
  coord.join(threads)