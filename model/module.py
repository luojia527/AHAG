#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 2019

@author: c503
"""

import numpy as np
import tensorflow as tf
import math


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)




def multihead_attention(queries, keys,
                        num_units, num_heads,causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        que_size=queries.get_shape().as_list()[1]


        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        dense_q = tf.layers.dense(queries, num_units, activation=tf.nn.relu, name="dense_q")
        dense_k = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_k")
        dense_v = tf.layers.dense(keys, num_units, activation=tf.nn.relu, name="dense_v")

        # Split and concat
        Q_ = tf.concat(tf.split(dense_q, int(num_heads/2), axis=2), axis=0)
        K_ = tf.concat(tf.split(dense_k, int(num_heads/2), axis=2), axis=0)
        V_ = tf.concat(tf.split(dense_k, int(num_heads/2), axis=2), axis=0)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))


        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)


        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [int(num_heads/2), 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        # Activation
        outputs = tf.nn.softmax(outputs)


        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [int(num_heads/2), 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks  #

        # Weighted sum
        outputs = tf.matmul(outputs, V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, int(num_heads/2), axis=0), axis=2)


        outputs += queries

        # Normalize
        atten_out_position = normalize(outputs)

        atten_out_position += positional_encoding(atten_out_position, que_size)

        atten_out_position = ff(atten_out_position, num_units=[num_units, num_units])

    return atten_out_position

def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def positional_encoding(inputs, maxlen, masking=True, scope="positional_encoding"):
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)
        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)
       # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
        return tf.to_float(outputs)

def ff(inputs, num_units, scope="positionwise_feedforward"):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.tanh)#relu
        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])
        # Residual connection
        outputs += inputs
        # Normalize
        outputs = normalize(outputs)

    return outputs


def self_convolution_context_pooling(self_feature_embedding, embedding_size, num_filters, scope="self_convolution_context_pooling"):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        filter_shape = [1, embedding_size, 1, num_filters]
        W_attention = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_attention")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv_attention_feature = tf.nn.conv2d(
            self_feature_embedding,
            W_attention,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_attention_feature")

        conv_context_feature = tf.nn.tanh(conv_attention_feature, name="tanh")
        pooled_context = tf.nn.max_pool(
            conv_context_feature,
            ksize=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pooled_context")


    return pooled_context

def self_convolution_context(feature_embedding, embedding_size, num_filters, scope="self_convolution_context"):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        filter_shape = [1, embedding_size, 1, num_filters]
        W_attention = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_attention")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv_attention_feature = tf.nn.conv2d(
            feature_embedding,
            W_attention,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_attention_feature")

    return conv_attention_feature

def self_convolution_context_021(feature_embedding, embedding_size, num_filters, scope="self_convolution_context"):
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        filter_shape = [1, embedding_size, 1, num_filters]
        W_attention = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_attention")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv_attention_feature_021 = tf.nn.conv2d(
            feature_embedding,
            W_attention,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_attention_feature")
    return conv_attention_feature_021

def optimized_trilinear_for_attention(args, user_len, item_len, input_keep_prob=1.0,
    scope='efficient_trilinear', bias_initializer=tf.zeros_initializer(),
    kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):

        weights4arg0 = tf.Variable(tf.random_uniform([arg_size, 1], -0.1, 0.1),
            name="linear_kernel4arg0")
        weights4arg1 = tf.Variable(tf.random_uniform([arg_size, 1], -0.1, 0.1),
            name="linear_kernel4arg1")
        weights4mlu = tf.Variable(tf.random_uniform([1, 1, arg_size], -0.1, 0.1),
            name="linear_kernel4mul")

        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, item_len])
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, user_len, 1])
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res =  subres0 + subres1 +subres2

        return res,subres0,subres1,subres2
def _linear(args,
            output_size,
            bias,
            bias_initializer=tf.zeros_initializer(),
            scope = None,
            kernel_initializer=initializer(),
            reuse = None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope, reuse = reuse) as outer_scope:
    weights = tf.get_variable(
        "linear_kernel", [total_arg_size, output_size],
        dtype=dtype,
        regularizer=regularizer,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "linear_bias", [output_size],
          dtype=dtype,
          regularizer=regularizer,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None
def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.

    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.

    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.

    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out



def convolution_feature(input_feature, filter_sizes,embedding_size,num_filters, input_length, scope="avgpool_convolution"):
    with tf.variable_scope(scope):
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    input_feature,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")

                # Maxpooling over the outputs
                pooled = tf.nn.avg_pool(  # avg_pool
                    h,
                    ksize=[1, input_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat

def fusion_gating_net(u_feature, uid_feature, n_latent, scope="fusion_gating_net"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        W_gating = tf.get_variable("W_gating", shape=[n_latent, n_latent],
                                  initializer=tf.contrib.layers.xavier_initializer())
        W_gate = tf.get_variable("W_gate", shape=[n_latent, n_latent],
                                  initializer=tf.contrib.layers.xavier_initializer())
        W_vgate = tf.get_variable("W_vgate", shape=[n_latent, n_latent],
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_gate = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="b_gate")

        # feature gating
        feature_rate = tf.sigmoid(tf.matmul((tf.add(u_feature, uid_feature)),W_gating)+b_gate)
        fusion_fea = tf.multiply((1 - feature_rate), uid_feature) + tf.multiply(feature_rate, u_feature)
        fuse_feature = tf.sigmoid(tf.matmul(uid_feature,W_gate)+tf.matmul(fusion_fea,W_vgate))
        fusion_feature = fuse_feature*tf.tanh(tf.matmul(fusion_fea,W_vgate)+b_gate)
        feature_enhanced = uid_feature+fusion_feature


    return feature_enhanced



def gating_network(u_feas, i_feas,n_latent, scope="gating_network"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        W_igate = tf.get_variable("W_igate", shape=[n_latent, n_latent],
                                  initializer=tf.contrib.layers.xavier_initializer())
        W_ugate = tf.get_variable("W_ugate", shape=[n_latent, n_latent],
                                  initializer=tf.contrib.layers.xavier_initializer())
        bi_gate = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi_gate")

        # feature gating
        feature_gate = tf.sigmoid(tf.matmul(u_feas, W_ugate) + tf.matmul(i_feas, W_igate) + bi_gate)
        i_feature_gate = i_feas * feature_gate


        # instance gating

        W_iinstance = tf.get_variable("W_iinstance", shape=[n_latent, n_latent],
                                      initializer=tf.contrib.layers.xavier_initializer())
        W_uinstance = tf.get_variable("W_uinstance", shape=[n_latent, n_latent],
                                      initializer=tf.contrib.layers.xavier_initializer())

        instance_gate = tf.sigmoid(tf.matmul(i_feature_gate, W_iinstance) + tf.matmul(u_feas, W_uinstance))
        item_feature_instance = i_feature_gate * instance_gate


    return i_feature_gate, item_feature_instance

