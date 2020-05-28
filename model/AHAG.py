#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 2019

@author: c503
"""

from typing import Any, Union, Tuple

import tensorflow as tf
from tensorflow import Operation
from module import multihead_attention, convolution_feature, gating_network,self_convolution_context_pooling,\
    self_convolution_context,fusion_gating_net,optimized_trilinear_for_attention



class AHAG(object):
    def __init__(
            self, user_length,item_length, user_vocab_size,item_vocab_size,fm_k,n_latent,user_num,item_num,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,l2_reg_V=0):
        self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None,1],name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.input_niid = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="input_niid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        self.input_u_mask = tf.cast(self.input_u, tf.bool)
        self.input_i_mask = tf.cast(self.input_i, tf.bool)
        self.input_u_leng = tf.reduce_sum(tf.cast(self.input_u_mask, tf.int32), axis=1)
        self.input_i_leng = tf.reduce_sum(tf.cast(self.input_i_mask, tf.int32), axis=1)
        self.input_u_maxlen = tf.reduce_max(self.input_u_leng)
        self.input_i_maxlen = tf.reduce_max(self.input_i_leng)
        self.input_u_mask = tf.slice(self.input_u_mask, [0, 0], [-1, self.input_u_maxlen])
        self.input_i_mask = tf.slice(self.input_i_mask, [0, 0], [-1, self.input_i_maxlen])



        with  tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -0.1, 0.1),
                name="W1")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)
            self.embedded_users_pad = tf.pad(self.embedded_user, [[0, 0], [2, 2], [0, 0]])
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, n_latent], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.uid = tf.reshape(self.uid, [-1, n_latent])

        with tf.variable_scope("self_user_attention"):

            self.self_attn_user = multihead_attention(self.embedded_user, self.embedded_user,
                                                      num_units=embedding_size, num_heads=8)

            self.self_attn_user_expanded_01 = tf.expand_dims(self.self_attn_user, -1)


            self.self_attn_user_expanded = self.self_attn_user_expanded_01
            user_feature_01 = self_convolution_context(self.self_attn_user_expanded, embedding_size, num_filters)
            user_feature_01=tf.transpose(user_feature_01, perm=[0, 1, 3, 2])
            user_feature_01 = tf.nn.relu(user_feature_01, name="relu")

            a_user_feature_02 = self_convolution_context(user_feature_01, embedding_size, num_filters)
            user_feature_02 = tf.transpose(a_user_feature_02, perm=[0, 1, 3, 2])
            user_feature_02 = tf.nn.sigmoid(user_feature_02, name="sigmoid")

            user_self_feature = tf.multiply(self.self_attn_user_expanded, user_feature_02)



            self.conv_user_weight = tf.reshape(user_self_feature, [-1, user_length])
            self.user_self_squeeze = tf.squeeze(user_self_feature,[3])

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -0.1, 0.1),
                name="W2")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)
            self.embedded_items_pad = tf.pad(self.embedded_item, [[0, 0], [2, 2], [0, 0]])

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, n_latent], -0.1, 0.1), name="iidmf")  #
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.iid = tf.reshape(self.iid, [-1, n_latent])

        with tf.variable_scope("self_item_attention"):

            self.self_attn_item = multihead_attention(self.embedded_item, self.embedded_item,
                                                      num_units=embedding_size, num_heads=8)

            self.self_attn_item_expanded_01 = tf.expand_dims(self.self_attn_item, -1)
            self.self_attn_item_expanded = self.self_attn_item_expanded_01
            item_feature_01 = self_convolution_context(self.self_attn_item_expanded, embedding_size, num_filters)
            item_feature_01 = tf.transpose(item_feature_01, perm=[0, 1, 3, 2])
            item_feature_01 = tf.nn.relu(item_feature_01, name="relu")

            a_item_feature_02 = self_convolution_context(item_feature_01, embedding_size, num_filters)

            item_feature_02 = tf.transpose(a_item_feature_02, perm=[0, 1, 3, 2])
            item_feature_02 = tf.nn.sigmoid(item_feature_02, name="sigmoid")

            item_self_feature = tf.multiply(self.self_attn_item_expanded, item_feature_02)
            self.conv_item_weight = tf.reshape(item_self_feature, [-1, item_length])
            self.item_self_squeeze = tf.squeeze(item_self_feature,[3])


        with tf.variable_scope("trilinear_matrix"):


            S,subres0,subres1,subres2 = optimized_trilinear_for_attention([self.user_self_squeeze, self.item_self_squeeze], user_length, item_length,
                                                  input_keep_prob=0.8)

            left_attention_01, right_attention_01 = tf.nn.softmax(tf.reduce_mean(tf.nn.tanh(S), axis=2)), tf.nn.softmax(tf.reduce_mean(tf.nn.tanh(S), axis=1))


            left_attention_01 = tf.expand_dims(tf.expand_dims(left_attention_01, -1), -1)

            right_attention_01 = tf.expand_dims(tf.expand_dims(right_attention_01, -1), -1)
            user_pooling = user_self_feature*left_attention_01
            item_pooling = item_self_feature*right_attention_01

            self.left_attention = tf.reshape(user_pooling, [-1, user_length])
            self.right_attention = tf.reshape(item_pooling, [-1, item_length])


            h_pool_flat_u = convolution_feature(user_pooling, filter_sizes, embedding_size, num_filters,
                                                user_length)
            h_pool_flat_i = convolution_feature(item_pooling, filter_sizes, embedding_size, num_filters,
                                                item_length)



        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(h_pool_flat_i, 1.0)

        with tf.variable_scope("get_fea", reuse=tf.AUTO_REUSE):
            num_filters_total = num_filters * len(filter_sizes)
            Wuu = tf.get_variable(
                "Wuu",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_feas_review = tf.nn.tanh(tf.matmul(self.h_drop_u, Wuu) + bu)


            Wii = tf.get_variable(
                "Wii",
                shape=[num_filters_total, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_feas_review = tf.nn.tanh(tf.matmul(self.h_drop_i, Wii) + bi)

        with tf.name_scope("gating_network"):

            #only fusion gate
            self.u_feas = fusion_gating_net(self.u_feas_review, self.uid, n_latent)
            self.item_first_filter = fusion_gating_net(self.i_feas_review, self.iid,n_latent)


            self.i_feas, self.item_fea_instance = gating_network(self.u_feas, self.item_first_filter, n_latent)



        with tf.name_scope('fm'):
            self.FM = tf.concat([self.u_feas, self.i_feas], 1)
            self.z = self.FM

            WF1 = tf.Variable(
                tf.random_uniform([n_latent * 2, 1], -0.1, 0.1), name='fm1')
            Wf2 = tf.Variable(
                tf.random_uniform([n_latent * 2, fm_k], -0.1, 0.1), name='fm2')

            one = tf.matmul(self.z, WF1)
            inte1 = tf.matmul(self.z, Wf2)
            inte2 = tf.matmul(tf.square(self.z), tf.square(Wf2))

            inter = (tf.square(inte1) - inte2) * 0.5

            # ________ Deep Layers __________
            inter_w = tf.Variable(tf.random_uniform([fm_k, fm_k], -0.1, 0.1), name='fm1')
            b_inter = tf.Variable(tf.constant(0.1, shape=[fm_k]), name="b_inter")
            inter_one = tf.matmul(inter, inter_w) + b_inter  # None * layer[i] * 1
            inter_two = tf.nn.relu(inter_one)
            inter_two = tf.nn.dropout(inter_two, self.dropout_keep_prob)  # dropout at each Deep layer
            inter_two = tf.matmul(inter_two, inter_w)  # None * 1

            inter_last = tf.reduce_sum(inter_two, 1, keep_dims=True)
            b = tf.Variable(tf.constant(0.1), name='bias')
            self.predictions = one + inter_last + b



        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
