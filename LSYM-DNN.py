#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:52:08 2020

@author: ziangcui
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import ray
from ray import tune

tf.reset_default_graph()

ray.init()

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, lr, nn_layer):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.lr = lr
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys')
            self.keep_prob_lstm = tf.placeholder(tf.float32, name='kpl')
            self.keep_prob_nn = tf.placeholder(tf.float32, name='kpn')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.variable_scope('out1'):
            self.add_nn_layer(inputs=self.lstm_pred, in_size=self.n_steps, out_size=30, activation_function=tf.nn.relu)
        for i in range(nn_layer - 1):
            with tf.variable_scope(str(i)):
                self.add_nn_layer(inputs=self.pred, in_size=30, out_size=30, activation_function=tf.nn.relu)
        with tf.variable_scope('out3'):
            self.add_nn_layer(inputs=self.pred, in_size=30, out_size=1, activation_function=None)
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def add_input_layer(self, ):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=self.keep_prob_lstm)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.lstm_pred = tf.matmul(l_out_x, Ws_out) + bs_out
            self.lstm_pred = tf.reshape(self.lstm_pred, [-1, self.n_steps])

    def add_nn_layer(self, inputs, in_size, out_size, activation_function=None):
        # tf.sqrt(2/in_size)是梯度爆炸/消失的处理方法
        # inputsnn_in = tf.reshape(self.lstm_pred,[-1,in_size])
        Weights = tf.Variable(tf.random_normal([in_size, out_size]) * tf.sqrt(2 / in_size))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
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
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

def train_func(config,reporter):
    TIME_STEPS = config["TIME_STEPS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    INPUT_SIZE = 13
    OUTPUT_SIZE = 1
    CELL_SIZE = config["CELL_SIZE"]
    LR = config["LR"]
    k1=0
    k2=0
    N_ITER = config["N_ITER"]
    plot_train_k = 3
    KEEP_PROB_LSTM = config["KEEP_PROB_LSTM"]
    KEEP_PROB_NN = config["KEEP_PROB_NN"]
    NN_LAYER = config["NN_LAYER"]

    is_train = True
    is_test = True

    savingPath = "/Users/ziangcui/Desktop/test/modle.ckpt"
    restorePath = "/Users/ziangcui/Desktop/test/"

    data = pd.read_csv("/Users/ziangcui/Desktop/同花顺/tonghuashun1.csv")
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




    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, LR, NN_LAYER)
    sess = tf.Session()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(init)


    t = 0
    max_cost = 10
    test_predict = []
    train_pred = []
    for iteration in range(N_ITER):
        k1=0
        cost_avg = 0
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
            cost_avg+=cost
        reporter(mean_cost = (100-cost_avg))


        if iteration%20==0:
            print('cost out: ', cost)

        if cost < max_cost:
            max_cost = cost
            save_path = saver.save(sess,savingPath)


    print("**********************")




all_trials = tune.run(
    train_func,
    name="quick start",
    stop={"mean_cost":100},
    config={"BATCH_SIZE": tune.grid_search([60]),
            "TIME_STEPS": tune.grid_search([30]),
            "CELL_SIZE": tune.grid_search([30]),
            "LR":tune.grid_search([0.003]),
            "N_ITER":tune.grid_search([5]),
            "KEEP_PROB_LSTM":tune.grid_search([0.6]),
            "KEEP_PROB_NN":tune.grid_search([0.8, 0.6]),
            "NN_LAYER":tune.grid_search([2])
}
)

best_config = all_trials.get_best_config(metric="mean_cost")
print(best_config)
print("****")