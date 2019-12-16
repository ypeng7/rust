import os

import tensorflow as tf

if tf.__version__.startswith("2"):
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.float32, name="x")
    y = tf.compat.v1.placeholder(tf.float32, name="y")

    w = tf.Variable(tf.compat.v1.random_uniform([1], -1.0, 1.0), name="w")
    b = tf.Variable(tf.zeros([1]), name="b")
    y_hat = w * x + b

    loss = tf.reduce_mean(tf.square(y_hat - y))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss, name="train")

    init = tf.compat.v1.variables_initializer(
        tf.compat.v1.global_variables(), name="init"
    )
    # Declare saver ops
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    definition = tf.compat.v1.Session().graph_def
    directory = "examples/regression_checkpoint"
    tf.io.write_graph(definition, directory, "model.pb", as_text=False)

else:
    x = tf.placeholder(tf.float32, name="x")
    y = tf.placeholder(tf.float32, name="y")

    w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="w")
    b = tf.Variable(tf.zeros([1]), name="b")
    y_hat = w * x + b

    loss = tf.reduce_mean(tf.square(y_hat - y))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss, name="train")

    init = tf.variables_initializer(tf.global_variables(), name="init")

    # Declare saver ops
    saver = tf.train.Saver(tf.global_variables())

    definition = tf.Session().graph_def
    directory = "examples/regression_checkpoint"
    tf.train.write_graph(definition, directory, "model.pb", as_text=False)
