"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
The Cartpole example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.8.0
gym 0.10.5
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from src.env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly


OUTPUT_GRAPH = False
LOG_DIR = './log'
#N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 2
MAX_EP_STEP =20
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

map_size = 64
obj_num = 25
action_type = 5

env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),max_num=obj_num)
N_S = [map_size, map_size, 2]
N_A = action_type * obj_num


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, *N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, *N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            self.conv1 = tf.layers.conv2d(inputs=self.s,filters=64,kernel_size=[5, 5],strides=[2, 2],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1_")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,training=True,epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

            self.conv2_1 = tf.layers.conv2d(inputs=self.conv1_out,filters=64,kernel_size=[3, 3],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_1")

            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,training=True,epsilon=1e-5,
                                                                   name='batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")

            self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_out_1,filters=64,kernel_size=[1, 1],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_2")

            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,training=True,epsilon=1e-5,
                                                                   name='batch_norm2_2')

            #SE1
            self.se1_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv2_batchnorm_2)
            self.se1_2 = tf.keras.layers.Dense(self.se1_1.shape[1]//2)(self.se1_1)
            self.se1_3 = tf.keras.layers.Activation('relu')(self.se1_2)
            self.se1_4 = tf.keras.layers.Dense(self.se1_1.shape[1])(self.se1_3)
            self.se1_5 = tf.keras.layers.Activation('sigmoid')(self.se1_4)
            self.se1_6 = tf.keras.layers.Reshape((1, 1, self.se1_1.shape[1]))(self.se1_5)

            self.conv2_batchnorm_2_se = self.conv2_batchnorm_2 * self.se1_6


            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2_se + self.conv1_out, name="conv2_out_2")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out_2,filters=128,kernel_size=[3, 3],strides=[2, 2],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,training=True,epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")

            self.conv4_1 = tf.layers.conv2d(inputs=self.conv3_out,filters=128,kernel_size=[3, 3],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_1")

            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,training=True,epsilon=1e-5,
                                                                   name='batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")

            self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_out_1,filters=128,kernel_size=[1, 1],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_2")

            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,training=True,epsilon=1e-5,
                                                                   name='batch_norm4_2')


            #SE2
            self.se2_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv4_batchnorm_2)
            self.se2_2 = tf.keras.layers.Dense(self.se2_1.shape[1]//2)(self.se2_1)
            self.se2_3 = tf.keras.layers.Activation('relu')(self.se2_2)
            self.se2_4 = tf.keras.layers.Dense(self.se2_1.shape[1])(self.se2_3)
            self.se2_5 = tf.keras.layers.Activation('sigmoid')(self.se2_4)
            self.se2_6 = tf.keras.layers.Reshape((1, 1, self.se2_1.shape[1]))(self.se2_5)
            self.conv4_batchnorm_2_se = self.conv4_batchnorm_2 * self.se2_6


            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2_se + self.conv3_out, name="conv4_out_2")

            self.conv5 = tf.layers.conv2d(inputs=self.conv4_out_2,filters=256,kernel_size=[3, 3],strides=[2, 2],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv5")

            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,training=True,epsilon=1e-5,name='batch_norm5')

            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")

            self.conv6_1 = tf.layers.conv2d(inputs=self.conv5_out,filters=256,kernel_size=[3, 3],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_1")

            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,training=True,epsilon=1e-5,name='batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")

            self.conv6_2 = tf.layers.conv2d(inputs=self.conv6_out_1,filters=256,kernel_size=[1, 1],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_2")

            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,training=True,epsilon=1e-5,name='batch_norm6_2')


            #SE3
            self.se3_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv6_batchnorm_2)
            self.se3_2 = tf.keras.layers.Dense(self.se3_1.shape[1]//2)(self.se3_1)
            self.se3_3 = tf.keras.layers.Activation('relu')(self.se3_2)
            self.se3_4 = tf.keras.layers.Dense(self.se3_1.shape[1])(self.se3_3)
            self.se3_5 = tf.keras.layers.Activation('sigmoid')(self.se3_4)
            self.se3_6 = tf.keras.layers.Reshape((1, 1, self.se3_1.shape[1]))(self.se3_5)
            self.conv6_batchnorm_2_se = self.conv6_batchnorm_2 * self.se3_6

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2_se + self.conv5_out, name="conv6_out_2")

            #self.finish_tag_ = tf.reshape(self.finish_tag, [-1, obj_num])
            #self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            self.flatten = tf.contrib.layers.flatten(self.conv6_out_2)
            cell_size = 64
            s = tf.expand_dims(self.flatten, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation

            self.output_ = tf.layers.dense(inputs=cell_out,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=None,
                                           name="pre_output_internal")

            #action_branch
            self.conv_a1_out = tf.layers.dense(inputs=self.output_,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=N_A,
                                           activation=tf.nn.softmax,
                                           name="act_output_internal")
            self.flatten_a1 = tf.contrib.layers.flatten(self.conv_a1_out)
            l_a = tf.layers.dense(self.flatten_a1, 256, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            self.conv1 = tf.layers.conv2d(inputs=self.s,filters=64,kernel_size=[5, 5],strides=[2, 2],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1_")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,training=True,epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

            self.conv2_1 = tf.layers.conv2d(inputs=self.conv1_out,filters=64,kernel_size=[3, 3],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_1")

            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,training=True,epsilon=1e-5,
                                                                   name='batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")

            self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_out_1,filters=64,kernel_size=[1, 1],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_2")

            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,training=True,epsilon=1e-5,
                                                                   name='batch_norm2_2')

            #SE1
            self.se1_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv2_batchnorm_2)
            self.se1_2 = tf.keras.layers.Dense(self.se1_1.shape[1]//2)(self.se1_1)
            self.se1_3 = tf.keras.layers.Activation('relu')(self.se1_2)
            self.se1_4 = tf.keras.layers.Dense(self.se1_1.shape[1])(self.se1_3)
            self.se1_5 = tf.keras.layers.Activation('sigmoid')(self.se1_4)
            self.se1_6 = tf.keras.layers.Reshape((1, 1, self.se1_1.shape[1]))(self.se1_5)
            self.conv2_batchnorm_2_se = self.conv2_batchnorm_2 * self.se1_6

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2_se + self.conv1_out, name="conv2_out_2")

            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out_2,filters=128,kernel_size=[3, 3],strides=[2, 2],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,training=True,epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")

            self.conv4_1 = tf.layers.conv2d(inputs=self.conv3_out,filters=128,kernel_size=[3, 3],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_1")

            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,training=True,epsilon=1e-5,
                                                                   name='batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")

            self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_out_1,filters=128,kernel_size=[1, 1],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_2")

            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,training=True,epsilon=1e-5,
                                                                   name='batch_norm4_2')


            #SE2
            self.se2_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv4_batchnorm_2)
            self.se2_2 = tf.keras.layers.Dense(self.se2_1.shape[1]//2)(self.se2_1)
            self.se2_3 = tf.keras.layers.Activation('relu')(self.se2_2)
            self.se2_4 = tf.keras.layers.Dense(self.se2_1.shape[1])(self.se2_3)
            self.se2_5 = tf.keras.layers.Activation('sigmoid')(self.se2_4)
            self.se2_6 = tf.keras.layers.Reshape((1, 1, self.se2_1.shape[1]))(self.se2_5)
            self.conv4_batchnorm_2_se = self.conv4_batchnorm_2 * self.se2_6


            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2_se + self.conv3_out, name="conv4_out_2")

            self.conv5 = tf.layers.conv2d(inputs=self.conv4_out_2,filters=256,kernel_size=[3, 3],strides=[2, 2],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv5")

            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,training=True,epsilon=1e-5,name='batch_norm5')

            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")

            self.conv6_1 = tf.layers.conv2d(inputs=self.conv5_out,filters=256,kernel_size=[3, 3],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_1")

            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,training=True,epsilon=1e-5,name='batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")

            self.conv6_2 = tf.layers.conv2d(inputs=self.conv6_out_1,filters=256,kernel_size=[1, 1],strides=[1, 1],padding="SAME",kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_2")

            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,training=True,epsilon=1e-5,name='batch_norm6_2')


            #SE3
            self.se3_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv6_batchnorm_2)
            self.se3_2 = tf.keras.layers.Dense(self.se3_1.shape[1]//2)(self.se3_1)
            self.se3_3 = tf.keras.layers.Activation('relu')(self.se3_2)
            self.se3_4 = tf.keras.layers.Dense(self.se3_1.shape[1])(self.se3_3)
            self.se3_5 = tf.keras.layers.Activation('sigmoid')(self.se3_4)
            self.se3_6 = tf.keras.layers.Reshape((1, 1, self.se3_1.shape[1]))(self.se3_5)
            self.conv6_batchnorm_2_se = self.conv6_batchnorm_2 * self.se3_6

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2_se + self.conv5_out, name="conv6_out_2")

            #self.finish_tag_ = tf.reshape(self.finish_tag, [-1, obj_num])
            #self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)

            self.flatten = tf.contrib.layers.flatten(self.conv6_out_2)


            cell_size = 64
            s = tf.expand_dims(self.flatten, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation

            self.output_ = tf.layers.dense(inputs=cell_out,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=None,
                                           name="pre_output_internal")
            l_c = tf.layers.dense(self.output_, 128, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):  # run by a local
        s = s[np.newaxis, :]
        prob_weights, cell_state = SESS.run([self.a_prob, self.final_state], {self.s: s, self.init_state: cell_state})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, cell_state


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),max_num=obj_num)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            #s = self.env.reset()

            current_num = np.random.randint(obj_num)+1
            self.env.randominit_crowded(current_num)
            s = self.env.getstate_3()
            ep_r = 0
            rnn_state = SESS.run(self.AC.init_state)    # zero rnn state at beginning
            keep_state = rnn_state.copy()       # keep rnn state for updating global net
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a,rnn_state_ = self.AC.choose_action(s,rnn_state)
                #s_, r, done, info = self.env.step(a)

                choice_index = int(a / action_type)
                choice_action = a % action_type
                r, done = self.env.move(choice_index, choice_action)
                s_ = self.env.getstate_3()

                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append([s])
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,

                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()   # replace the keep_state as the new initial rnn state_


                s = s_
                rnn_state = rnn_state_  # renew rnn state
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()