import tensorflow as tf

if tf.__version__.startswith("2"):
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder(tf.int32, name="x")
    y = tf.compat.v1.placeholder(tf.int32, name="y")
    z = tf.add(x, y, name="z")

    tf.compat.v1.variables_initializer(tf.compat.v1.global_variables(), name="init")

    definition = tf.compat.v1.Session().graph_def
    directory = "examples/addition"
    tf.io.write_graph(definition, directory, "model.pb", as_text=False)
else:
    x = tf.placeholder(tf.int32, name="x")
    y = tf.placeholder(tf.int32, name="y")
    z = tf.add(x, y, name="z")

    tf.variables_initializer(tf.global_variables(), name="init")

    definition = tf.Session().graph_def
    directory = "examples/addition"
    tf.train.write_graph(definition, directory, "model.pb", as_text=False)
