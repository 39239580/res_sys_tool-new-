import tensorflow as tf
bias = tf.Variable(tf.constant(0.01), dtype=tf.float32)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(bias)
    print(sess.run(bias))
