#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:57:52 2020

@author: ziangcui
"""


import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import math



step = 0

def fun(TIME_STEPS = 20,
    BATCH_SIZE = 50,
    INPUT_SIZE = 13,
    OUTPUT_SIZE = 1,
    CELL_SIZE = 14,
    LSTM_LAYER = 7,
    LR = 0.0006,
    k1=0,
    k2=0,
    N_ITER = 100,
    plot_train_k = 3,
    KEEP_PROB_LSTM = 0.6,
    KEEP_PROB_NN = 0.5,
    NN_LAYER = 2,
    is_train = True,
    is_test = True,
    savingPath = "/Users/ziangcui/Desktop/test/modle.ckpt",
    restorePath = "/Users/ziangcui/Desktop/test/",
    data = pd.read_csv("/Users/ziangcui/Desktop/pycharm工作空间/dataset/tonghuashun1.csv")):

    tf.reset_default_graph()

    data = np.array(data)
    n1 = len(data[0])-1
    train_end_index = math.floor(len(data)*0.9)
    data_train = data[0:train_end_index]
    data_test = data[train_end_index+1:]

    mean_train = np.mean(data_train,axis=0)
    std_train = np.std(data_train, axis=0)
    mean_test = np.mean(data_test,axis=0)
    std_test = np.std(data_test,axis=0)

    INPUT_SIZE = n1

    train_data = []
    train_target = []
    batch_index = []

    data_train = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
    data_test = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)

    for i in range(len(data_train) - TIME_STEPS):
        if i % BATCH_SIZE == 0:
            batch_index.append(i)

        x = data_train[i:i + TIME_STEPS, :n1]
        # y存储label
        y = data_train[i + TIME_STEPS-1, n1]
        # np.newaxis分别是在行或列上增加维度
        train_data.append(x.tolist())
        train_target.append(y)
    batch_index.append((len(data_train) - TIME_STEPS))

    test_data = []
    test_target = []

    for i in range(len(data_test) - TIME_STEPS):
        x = data_test[i:i + TIME_STEPS, :n1]
        y = data_test[TIME_STEPS+i-1, n1]
        test_data.append(x.tolist())
        test_target.append(y)

    def get_train_pic():
        k = len(train_data)//BATCH_SIZE - plot_train_k
        datat = data_train[k*BATCH_SIZE+TIME_STEPS-1:(k+plot_train_k)*BATCH_SIZE+TIME_STEPS-1,n1]
        return datat

    def get_batch():
        seq1 = np.array(train_data)
        seq1 = seq1[k1:k1+BATCH_SIZE,:,:]
        res1 = np.array(train_target)
        res1 = res1[k1:k1+BATCH_SIZE].reshape(-1,1)
        return [seq1, res1, batch_index]

    def get_test_data():
        datat = np.array(test_data)
        datat = datat[k2:k2+BATCH_SIZE,:,:]
        return [datat, test_target]



    class LSTMRNN(object):
        def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lstm_layer,nn_layer):
            self.n_steps = n_steps
            self.input_size = input_size
            self.output_size = output_size
            self.cell_size = cell_size
            self.batch_size = batch_size
            self.lstm_layer = lstm_layer
            with tf.name_scope('inputs'):
                self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
                self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
                self.keep_prob_lstm = tf.placeholder(tf.float32,name='kpl')
                self.keep_prob_nn = tf.placeholder(tf.float32,name='kpn')
            with tf.variable_scope('in_hidden'):
                self.add_input_layer()
            with tf.variable_scope('LSTM_cell'):
                self.add_cell()
            with tf.variable_scope('out_hidden'):
                self.add_output_layer()
            with tf.variable_scope('out1'):
                self.add_nn_layer(inputs=self.lstm_pred,in_size=self.n_steps,out_size=30,activation_function=tf.nn.relu)
            with tf.variable_scope('out2'):
                self.add_nn_layer(inputs=self.pred,in_size=30,out_size=30,activation_function=tf.nn.relu)
            with tf.variable_scope('out3'):
                self.add_nn_layer(inputs=self.pred,in_size=30,out_size=1,activation_function=None)
            with tf.name_scope('cost'):
                self.compute_cost()
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

        def add_input_layer(self,):
            l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
            # Ws (in_size, cell_size)
            Ws_in = self._weight_variable([self.input_size, self.cell_size])
            # bs (cell_size, )
            bs_in = self._bias_variable([self.cell_size,])
            # l_in_y = (batch * n_steps, cell_size)
            with tf.name_scope('Wx_plus_b'):
                l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            # reshape l_in_y ==> (batch, n_steps, cell_size)
            self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

        def add_cell(self):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*self.lstm_layer)
            cell= tf.contrib.rnn.DropoutWrapper(multi_layer_cell,input_keep_prob=self.keep_prob_lstm)
            with tf.name_scope('initial_state'):
                self.cell_init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
                cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

        def add_output_layer(self):
            # shape = (batch * steps, cell_size)
            l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
            Ws_out = self._weight_variable([self.cell_size, self.output_size])
            bs_out = self._bias_variable([self.output_size, ])
            # shape = (batch * steps, output_size)
            with tf.name_scope('Wx_plus_b'):
                self.lstm_pred = tf.matmul(l_out_x, Ws_out) + bs_out
                self.lstm_pred = tf.reshape(self.lstm_pred,[-1,self.n_steps])

        def add_nn_layer(self,inputs,in_size,out_size,activation_function=None):
            #tf.sqrt(2/in_size)是梯度爆炸/消失的处理方法
            #inputsnn_in = tf.reshape(self.lstm_pred,[-1,in_size])
            Weights = tf.Variable(tf.random_normal([in_size,out_size])*tf.sqrt(2/in_size))
            biases = tf.Variable(tf.zeros([1,out_size])) + 0.1
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, self.keep_prob_nn)
            if activation_function == None:
                self.pred = Wx_plus_b
            else:
                self.pred = activation_function(Wx_plus_b)


        def compute_cost(self):

            with tf.name_scope('average_cost'):
                self.cost = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])))


        @staticmethod
        def ms_error(labels, logits):
            return tf.square(tf.subtract(labels, logits))

        def _weight_variable(self, shape, name='weights'):
            initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
            return tf.get_variable(shape=shape, initializer=initializer, name=name)

        def _bias_variable(self, shape, name='biases'):
            initializer = tf.constant_initializer(0.1)
            return tf.get_variable(name=name, shape=shape, initializer=initializer)

    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LSTM_LAYER, NN_LAYER)
    sess = tf.Session()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init)

    if is_train:
        t = 0
        max_cost = 10
        cost_sum = []
        step = 0
        for iteration in range(N_ITER):
            k1=0
            step = step + 1

            for i in range(len(train_data)//BATCH_SIZE):
                seq, res , index= get_batch()
                k1 = k1 + BATCH_SIZE
                if iteration == 0:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.keep_prob_lstm:KEEP_PROB_LSTM,
                        model.keep_prob_nn:KEEP_PROB_NN
                        # create initial state
                    }
                else:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.keep_prob_lstm:KEEP_PROB_LSTM,
                        model.keep_prob_nn:KEEP_PROB_NN,
                        model.cell_init_state: state    # use last state as the initial state for this run
                    }

                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)

            cost_sum.append(cost)

            if iteration%20==0:
                print('cost out: ', cost)


            if cost < max_cost:
                max_cost = cost
                save_path = saver.save(sess,savingPath)



        print("**********************")

    if is_test:
        model_file = tf.train.latest_checkpoint(restorePath)
        saver.restore(sess,model_file)

        test_predict = []
        train_pred = []

        k1=0
        for i in range(len(train_data)//BATCH_SIZE):
            if i >= (len(train_data)//BATCH_SIZE - plot_train_k):
                seq, res , index = get_batch()

                feed_dict = {
                    model.xs: seq,
                    model.keep_prob_lstm:1,
                    model.keep_prob_nn:1
                }
                pred = sess.run(
                    model.pred,
                    feed_dict=feed_dict)

                pred = pred.reshape((-1))
                train_pred.extend(pred)

            k1 = k1 + BATCH_SIZE

        test_x, test_y = get_test_data()
        for step in range(len(test_data)//BATCH_SIZE):
            test_x, test_y = get_test_data()
            k2 = k2 + BATCH_SIZE
            feed_dict = {
                model.xs: test_x,
                model.keep_prob_lstm:1,
                model.keep_prob_nn:1
            }
            pred = sess.run(
                model.pred,
                feed_dict=feed_dict)
            pred = pred.reshape((-1))
            test_predict.extend(pred)

        test_y = test_y[:(len(test_data)//BATCH_SIZE)*BATCH_SIZE]
        print("finish")

        data2 = get_train_pic()
        data2 = np.array(data2) * std_train[n1] + mean_train[n1]
        train_pred = np.array(train_pred) * std_train[n1] + mean_train[n1]
        # 相对误差=（测量值-计算值）/计算值×100%
        acc = np.average(np.abs(train_pred - data2[:len(train_pred)]) / data2[:len(train_pred)])
        print("训练预测的相对误差:", acc)


        test_y = np.array(test_y) * std_test[n1] + mean_test[n1]
        test_predict = np.array(test_predict) * std_test[n1] + mean_test[n1]

        test_y = test_y.reshape(-1,1)
        test_predict = test_predict.reshape(-1,1)
        data2 = data2.reshape(-1,1)
        train_pred = train_pred.reshape(-1,1)

        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])
        print("测试预测的相对误差:", acc)

        test_predict = np.vstack((train_pred,test_predict))
        test_y = np.vstack((data2,test_y))

        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])
        print("总的相对误差:", acc)

        return [test_predict,test_y]

fun()