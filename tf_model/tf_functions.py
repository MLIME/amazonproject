import tensorflow as tf


def init_wb(shape, name, mean, std, random_seed):
    """
    Function initialize one matrix of weights and one bias vector.

    :type shape: tuple
    :type name: str
    :type random_seed: None or int
    :rtype: dictionary
    """
    Winit = tf.truncated_normal(shape, mean=mean, stddev=std, seed=random_seed)
    binit = tf.zeros(shape[-1])
    layer = {}
    layer["weights"] = tf.get_variable(name + "/weights",
                                       dtype=tf.float32,
                                       initializer=Winit)
    layer["bias"] = tf.get_variable(name + "/bias",
                                    dtype=tf.float32,
                                    initializer=binit)
    return layer


def apply_conv(input_tensor, layer):
    """
    Function that applies convolution in the input tensor using
    the layer["weights"] as filter.

    :type input_tensor: tf tensor
    :type layer: dictionary
    :rtype: tf tensor
    """
    weights = layer['weights']
    bias = layer['bias']
    conv_layer = tf.nn.conv2d(input=input_tensor,
                              filter=weights,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
    conv_layer += bias
    return conv_layer


def apply_pooling(input_layer):
    """
    Function that applies max pooling in the input tensor.

    :type input_tensor: tf tensor
    :rtype: tf tensor
    """
    pool_layer = tf.nn.max_pool(value=input_layer,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')
    pool_layer = tf.nn.relu(pool_layer)
    return pool_layer


def linear_activation(input_tensor, layer):
    """
    Function that applies the linear activation in the input tensor.

    :type input_tensor: tf tensor
    :type layer: dictionary
    :rtype: tf tensor
    """
    return tf.add(tf.matmul(input_tensor, layer['weights']),
                  layer['bias'])


def gd_train(loss,
             starter_learning_rate,
             steps_for_decay,
             decay_rate):
    """
    Function that return the optimizer to minimize the function loss.
    It uses the general step to change the learning rate.
    :type loss: tf tensor
    :type starter_learning_rate: float
    :type steps_for_decay: int
    :type decay_rate: float
    :rtype: tf optimizer
    """
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step,
                                               steps_for_decay,
                                               decay_rate,
                                               staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer.minimize(loss, global_step=global_step)