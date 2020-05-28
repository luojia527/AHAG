#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 2019

@author: c503
"""

import numpy as np
import tensorflow as tf
import datetime

import pickle
import AHAG
import os
import random

tf.flags.DEFINE_string("word2vec", "../data/glove.6B.100d.txt", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("valid_data","../data/music/music.valid", " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/music/music.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "../data/music/music.train", "Data for training")
tf.flags.DEFINE_string("test_data","../data/music/music.test", " Data for test")

# ==================================================

# Model Hyperparameters

tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 1, "L2 regularizaion lambda")
tf.flags.DEFINE_float("l2_reg_V", 0.0, "L2 regularizaion V")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, uid, iid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()


    return accuracy, mae


def dev_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0#0.5
    }
    step, loss, accuracy, mae, conv_user_weight, conv_item_weight, left_attention, right_attention = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae, deep.conv_user_weight, deep.conv_item_weight, deep.left_attention, deep.right_attention],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    return [loss, accuracy, mae, conv_user_weight, conv_item_weight, left_attention, right_attention]

def init_W():
    print("Load word2vec file {}\n".format(FLAGS.word2vec))
    word2vec_dic = {}
    index2word = {}
    mean = np.zeros(FLAGS.embedding_dim)
    count = 0
    with open(FLAGS.word2vec, "rb") as f:
        for line in f:
            values = line.split()
            word = values[0]
            word_vec = np.array(values[1:], dtype='float32')
            word2vec_dic[word] = word_vec
            mean = mean + word_vec
            index2word[count] = word
            count = count + 1
        mean = mean / count


    initW_u = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
    for word_u in vocabulary_user:
        if word_u in word2vec_dic:
            initW_u[vocabulary_user[word_u]] = word2vec_dic[word_u]
        else:
            initW_u[vocabulary_user[word_u]] = np.random.normal(mean, 0.1, size=FLAGS.embedding_dim)

    initW_i = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
    for word_i in vocabulary_item:
        if word_i in word2vec_dic:
            initW_i[vocabulary_item[word_i]] = word2vec_dic[word_i]
        else:
            initW_i[vocabulary_item[word_i]] = np.random.normal(mean, 0.1, size=FLAGS.embedding_dim)

    return initW_u, initW_i

def get_para(para_file):
    print("Loading data....")
    pkl_file = open(FLAGS.para_data, 'rb')

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    user_length = para['user_length']
    item_length = para['item_length']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    valid_length = para['valid_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']
    user_voc = para['user_voc']
    item_voc = para['item_voc']

    return user_num, item_num, user_length, item_length, vocabulary_item, vocabulary_user, train_length, \
           valid_length, test_length, u_text, i_text, user_voc, item_voc

if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))

    print("Loading data...")
    user_num, item_num, user_length, item_length, vocabulary_item, vocabulary_user, train_length, \
    valid_length, test_length, u_text, i_text, user_voc, item_voc = get_para(FLAGS.para_data)

    np.random.seed(2019)
    random_seed = 2019

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = AHAG.AHAG(
                user_num=user_num,
                item_num=item_num,
                user_length=user_length,
                item_length=item_length,
                user_vocab_size=len(vocabulary_user),
                item_vocab_size=len(vocabulary_item),
                embedding_size=FLAGS.embedding_dim,
                fm_k=16,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                l2_reg_V=FLAGS.l2_reg_V,
                n_latent=16)
            tf.set_random_seed(random_seed)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            optimizer = tf.train.AdamOptimizer(0.00003, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            train_op = optimizer

            sess.run(tf.global_variables_initializer())

            if FLAGS.word2vec:
                initW_u, initW_i = init_W()
                sess.run(deep.W1.assign(initW_u))                
                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))
                sess.run(deep.W2.assign(initW_i)) 


            l = (train_length / FLAGS.batch_size) + 1
            ll = 0
            epoch = 1
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0

            pkl_file = open(FLAGS.train_data, 'rb')
            train_data = pickle.load(pkl_file)
            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open(FLAGS.valid_data, 'rb')
            valid_data = pickle.load(pkl_file)
            valid_data = np.array(valid_data)
            pkl_file.close()

            pkl_file = open(FLAGS.test_data, 'rb')
            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()            
            
            data_size_train = len(train_data)
            data_size_valid = len(valid_data)            
            data_size_test = len(test_data)
            batch_size = 100
            ll = int(len(train_data) / batch_size)+1

            print('Stating epoch training')
            for epoch in range(100):
                # Shuffle the data at each epoch

                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, y_batch = zip(*data_train)

                    u_batch = []
                    i_batch = []

                    for i in range(len(uid)):
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)

                    t_rmse, t_mae = train_step(u_batch, i_batch, uid, iid, y_batch, batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae

                    if batch_num % 1000 == 0 and batch_num > 1:
                        print("\nEvaluation:")
                        print (batch_num)
                        loss_valid = 0
                        accuracy_valid = 0
                        mae_valid = 0
                        ll_valid = int(len(valid_data) / batch_size) + 1
                        for batch_num2 in range(ll_valid):
                            start_index = batch_num2 * batch_size
                            end_index = min((batch_num2 + 1) * batch_size, data_size_valid)
                            data_valid = valid_data[start_index: end_index]

                            userid_valid, itemid_valid, y_valid = zip(*data_valid)


                            u_valid = []
                            i_valid = []
                            for i in range(len(userid_valid)):
                                u_valid.append(u_text[userid_valid[i][0]])
                                i_valid.append(i_text[itemid_valid[i][0]])
                            u_valid = np.array(u_valid)
                            i_valid = np.array(i_valid)

                            loss, accuracy, mae, conv_user_weight, conv_item_weight, left_attention, right_attention = dev_step(
                                u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                            loss_valid = loss_valid + len(u_valid) * loss
                            accuracy_valid = accuracy_valid + len(u_valid) * accuracy

                            mae_valid = mae_valid + len(u_valid) * mae
                        print ("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_valid / valid_length,accuracy_valid / valid_length, mae_valid / valid_length))




                print (str(epoch) + ':\n')
                print("\nEvaluation:")
                print ("train:rmse,mae:", train_rmse / ll, train_mae / ll)
                train_rmse = 0
                train_mae = 0

                loss_test = 0
                accuracy_test = 0
                mae_test = 0

                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    userid_test, itemid_test, y_test = zip(*data_test)

                    u_test = []
                    i_test = []
                    u_word_test = []
                    i_word_test = []

                    for i in range(len(userid_test)):
                        u_test.append(u_text[userid_test[i][0]])

                        u_word_test.append(user_voc[i])

                    u_test = np.array(u_test)

                    for j in range(len(itemid_test)):
                        i_test.append(i_text[itemid_test[j][0]])
                        i_word_test.append(item_voc[j])
                    i_test = np.array(i_test)

                    loss, accuracy, mae, conv_user_weight, conv_item_weight, left_attention, right_attention = dev_step(
                        u_test, i_test, userid_test, itemid_test, y_test)

                    loss_test = loss_test + len(
                        u_test) * loss
                    accuracy_test = accuracy_test + len(u_test) * accuracy
                    mae_test = mae_test + len(u_test) * mae


                print ("loss_test {:g}, rmse_test {:g}, mae_test {:g}".format(loss_test / test_length,accuracy_test / test_length, mae_test / test_length))




                if (epoch == 0):
                    fh = open('result.txt', 'w')
                else:
                    fh = open('result.txt', 'a')
                rmse = accuracy_test / test_length
                mae = mae_test / test_length
                fh.write("rmse " + str(rmse))
                fh.write("mae " + str(mae))
                fh.close()
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae

            fh = open('result.txt', 'a')
            fh.write(str(best_rmse)+"\n")
            fh.write(str(best_mae)+"\n")
            print('best rmse:', best_rmse)
            print('best mae:', best_mae)



    print('end')
