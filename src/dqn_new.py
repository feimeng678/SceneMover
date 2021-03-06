import tensorflow as tf
import numpy as np
import os
from env import ENV
from env import ENV3
from env import ENV_M
from env import ENV_M_C
from env import ENV_M_C_5
from env import ENV_scene
from env import ENV_M_C_L
from env import ENV_scene_new_action
from env import ENV_scene_new_action_pre_state
from env import ENV_scene_new_action_pre_state_penalty
from env import ENV_scene_new_action_pre_state_penalty_conflict
from env import ENV_scene_new_action_pre_state_penalty_conflict_easy
from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic
from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max
from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape
from copy import deepcopy
from collections import deque
import time
import pickle

from mythread import MyThread
import threading



class DQNetwork:
    def __init__(self, state_size=[5,5], action_size=3, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # # Input is 84x84x4
            # self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
            #                              filters = 32,
            #                              kernel_size = [8,8],
            #                              strides = [4,4],
            #                              padding = "VALID",
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                              name = "conv1")
            
            # self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm1')
            
            # self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # ## --> [20, 20, 32]
            
            
            # """
            # Second convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
            #                      filters = 64,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv2")
        
            # self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm2')

            # self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # ## --> [9, 9, 64]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.contrib.layers.flatten(self.inputs_)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 256,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 512,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")
            
            
            self.output = tf.layers.dense(inputs = self.fc2, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class DQNetwork2:
    def __init__(self, state_size=[5,5], action_size=3, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # # Input is 84x84x4
            # self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
            #                              filters = 32,
            #                              kernel_size = [8,8],
            #                              strides = [4,4],
            #                              padding = "VALID",
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                              name = "conv1")
            
            # self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm1')
            
            # self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # ## --> [20, 20, 32]
            
            
            # """
            # Second convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
            #                      filters = 64,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv2")
        
            # self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm2')

            # self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # ## --> [9, 9, 64]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.contrib.layers.flatten(self.inputs_)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class DQNetwork3:
    def __init__(self, state_size=[5,5,4], action_size=3, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # # Input is 84x84x4
            # self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
            #                              filters = 32,
            #                              kernel_size = [8,8],
            #                              strides = [4,4],
            #                              padding = "VALID",
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                              name = "conv1")
            
            # self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm1')
            
            # self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            # ## --> [20, 20, 32]
            
            
            # """
            # Second convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
            #                      filters = 64,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv2")
        
            # self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm2')

            # self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            # ## --> [9, 9, 64]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.contrib.layers.flatten(self.inputs_)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork4:
    def __init__(self, state_size=[5,5,4], action_size=3, frame_num=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            self.action_chain = tf.placeholder(tf.float32, [None, action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out), self.action_chain], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork5:
    def __init__(self, state_size=[5,5,4], action_size=3, frame_num=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            self.action_chain = tf.placeholder(tf.float32, [None, action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out), self.action_chain], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork6:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, frame_num=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out), self.action_chain], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork7:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, frame_num=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag, self.action_chain], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork8:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork9:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out),
            # tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), 
            self.finish_tag], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork10:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            # self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and)], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            self.fc3 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc3, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork11:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [4, 4, 128]
            
            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv2_out),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            # self.fc2 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc2")

            # self.fc3 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc1, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork12:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 128,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 256,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 512,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 1024,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            # self.fc2 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc2")

            # self.fc3 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc1, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)
            print(self.output)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)

class DQNetwork13:
    def __init__(self, state_size=[5,5,4], action_space=5, num_objects=5, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_space*num_objects
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None, self.action_size], name="target")
            self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32,[None, num_objects], name="finish_tag")

            conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 128,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out',self.conv1_out)
            
            
            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_1")
        
            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")


            self.conv2_2 = tf.layers.conv2d(inputs = self.conv2_out_1,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2_2")
        
            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2+self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out',self.conv2_out_2)
            

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out_2,
                                         filters = 256,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')
            
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out',self.conv3_out)


            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs = self.conv3_out,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_1")
        
            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")


            self.conv4_2 = tf.layers.conv2d(inputs = self.conv4_out_1,
                                 filters = 256,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4_2")
        
            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2+self.conv3_out, name="conv4_out_2")
            print('conv4_out',self.conv4_out_2)
            ## --> [4, 4, 128]


            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs = self.conv4_out_2,
                                         filters = 512,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv5")
            
            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm5')
            
            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out',self.conv5_out)


            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs = self.conv5_out,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_1")
        
            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")


            self.conv6_2 = tf.layers.conv2d(inputs = self.conv6_out_1,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv6_2")
        
            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2+self.conv5_out, name="conv6_out_2")
            print('conv6_out',self.conv6_out_2)



            """
            Seventh convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv7 = tf.layers.conv2d(inputs = self.conv6_out_2,
                                         filters = 512,
                                         kernel_size = [4,4],
                                         strides = [2,2],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv7")
            
            self.conv7_batchnorm = tf.layers.batch_normalization(self.conv7,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm7')
            
            self.conv7_out = tf.nn.elu(self.conv7_batchnorm, name="conv7_out")
            print('conv7_out',self.conv7_out)


            """
            Eighth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv8_1 = tf.layers.conv2d(inputs = self.conv7_out,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv8_1")
        
            self.conv8_batchnorm_1 = tf.layers.batch_normalization(self.conv8_1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm8_1')

            self.conv8_out_1 = tf.nn.elu(self.conv8_batchnorm_1, name="conv8_out_1")


            self.conv8_2 = tf.layers.conv2d(inputs = self.conv8_out_1,
                                 filters = 512,
                                 kernel_size = [4,4],
                                 strides = [1,1],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv8_2")
        
            self.conv8_batchnorm_2 = tf.layers.batch_normalization(self.conv8_2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm8_2')

            self.conv8_out_2 = tf.nn.elu(self.conv8_batchnorm_2+self.conv7_out, name="conv8_out_2")
            print('conv8_out',self.conv8_out_2)

            
            # """
            # Third convnet:
            # CNN
            # BatchNormalization
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")
        
            # self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
            #                                        training = True,
            #                                        epsilon = 1e-5,
            #                                          name = 'batch_norm3')

            # self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # ## --> [3, 3, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv8_out_2),tf.contrib.layers.flatten(self.conflict_matrix), tf.contrib.layers.flatten(conflict_matrix_and), self.finish_tag], -1)
            ## --> [1152]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            # self.fc2 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc2")

            # self.fc3 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc1, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            # self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.Q = tf.multiply(self.output, self.actions_)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            temp = tf.square(self.target_Q - self.Q)
            loss_details = tf.reduce_mean(tf.reshape(temp,[-1,num_objects,action_space]),axis=[0,1])
            print(loss_details)
            self.loss_details = [loss_details[i] for i in range(action_space)]
            self.loss = tf.reduce_mean(temp)
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer2 = tf.train.RMSPropOptimizer(0.00005).minimize(self.loss)



class DQNetworkC:
    def __init__(self, state_size=[5,5,4], action_size=3, learning_rate=0.0002, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 128x128xnum
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [32, 32, 32]
            
            
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [4,4],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [8, 8, 64]
            
            
            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "SAME",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")
        
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [4, 4, 128]
            
            
            # self.flatten = tf.layers.flatten(self.inputs_)

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            ## --> [2048]
            
            
            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                    units = 1024,
                                    activation = tf.nn.elu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")

            # self.fc3 = tf.layers.dense(inputs = self.fc1,
            #                         units = 1024,
            #                         activation = tf.nn.elu,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="fc3")
            
            
            self.output = tf.layers.dense(inputs = self.fc2, 
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size, 
                                        activation=None)

  
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(buffer_size,
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

    def size(self):
        return len(self.buffer)

# """
# This function will do the part
# With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
# """
# def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
#     ## EPSILON GREEDY STRATEGY
#     # Choose action a from state s using epsilon greedy.
#     ## First we randomize a number
#     exp_exp_tradeoff = np.random.rand()

#     # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
#     explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
#     if (explore_probability > exp_exp_tradeoff):
#         # Make a random action (exploration)
#         action = random.choice(possible_actions)
        
#     else:
#         # Get action from Q-network (exploitation)
#         # Estimate the Qs values state
#         Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
#         # Take the biggest Q value (= the best action)
#         choice = np.argmax(Qs)
#         action = possible_actions[int(choice)]

#     return action, explore_probability
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

def train():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 5000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 3
    start_to_train = 100000
    state_size = [15,15]

    possible_actions = [(1,0,0),(0,1,0),(0,0,1)]

    tensorboard_path = "tensorboard/dqn/"
    weight_path = "weights"
    tensorboard_path = "tensorboard/dqn_3o_15_finetune/"
    weight_path = "weights_3o_15_finetune"
    tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer/"
    weight_path = "weights_3o_15_bigger_buffer"
    tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_finetune/"
    weight_path = "weights_3o_15_bigger_buffer_finetune"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork(state_size=state_size,learning_rate=learning_rate,action_size=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory()

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV3(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1000)
    
    finished = True
    for i in range(start_to_train):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        action_index = np.random.randint(action_space)
        action = np.zeros(action_space)
        action[action_index] = 1
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
    
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    for step in range(total_episodes):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space)
            action = np.zeros(3)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(3)
            action[choice] = 1
        
        action_index = np.argmax(action)
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)
        
        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train2():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 3
    start_to_train = 100000
    state_size = [15,15]

    possible_actions = [(1,0,0),(0,1,0),(0,0,1)]

    # tensorboard_path = "tensorboard/dqn/"
    # weight_path = "weights"
    # tensorboard_path = "tensorboard/dqn_3o_15_finetune/"
    # weight_path = "weights_3o_15_finetune"
    tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_v2/"
    weight_path = "weights_3o_15_bigger_buffer_v2"
    # tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_v2_finetune/"
    # weight_path = "weights_3o_15_bigger_buffer_v2_finetune"
    # tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_finetune/"
    # weight_path = "weights_3o_15_bigger_buffer_finetune"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork2(state_size=state_size,learning_rate=learning_rate,action_size=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory()

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_M(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1000)
    
    finished = True
    for i in range(start_to_train):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        action_index = np.random.randint(action_space)
        action = np.zeros(action_space)
        action[action_index] = 1
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
    
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    for step in range(total_episodes):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space)
            action = np.zeros(3)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(3)
            action[choice] = 1
        
        action_index = np.argmax(action)
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_M(size=(15,15))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate()
                Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                while True:
                    action = np.argmax(Qs)
                    reward, done = env_test.move(action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train3():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 3
    start_to_train = 100000
    state_size = [15,15,4]

    possible_actions = [(1,0,0),(0,1,0),(0,0,1)]

    # tensorboard_path = "tensorboard/dqn/"
    # weight_path = "weights"
    # tensorboard_path = "tensorboard/dqn_3o_15_finetune/"
    # weight_path = "weights_3o_15_finetune"
    tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_v3/"
    weight_path = "weights_3o_15_bigger_buffer_v3"
    # tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_v2_finetune/"
    # weight_path = "weights_3o_15_bigger_buffer_v2_finetune"
    # tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_finetune/"
    # weight_path = "weights_3o_15_bigger_buffer_finetune"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory()

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_M_C(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1000)
    
    finished = True
    for i in range(start_to_train):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        action_index = np.random.randint(action_space)
        action = np.zeros(action_space)
        action[action_index] = 1
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
    
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    for step in range(total_episodes):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space)
            action = np.zeros(3)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(3)
            action[choice] = 1
        
        action_index = np.argmax(action)
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_M_C(size=(15,15))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate()
                Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                while True:
                    action = np.argmax(Qs)
                    reward, done = env_test.move(action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train4():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 5
    start_to_train = 100000
    state_size = [15,15,6]

    # possible_actions = [(1,0,0),(0,1,0),(0,0,1)]

    # tensorboard_path = "tensorboard/dqn/"
    # weight_path = "weights"
    # tensorboard_path = "tensorboard/dqn_3o_15_finetune/"
    # weight_path = "weights_3o_15_finetune"
    tensorboard_path = "tensorboard/dqn_5o_15_bigger_buffer_v3/"
    weight_path = "weights_5o_15_bigger_buffer_v3"
    # tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_v2_finetune/"
    # weight_path = "weights_3o_15_bigger_buffer_v2_finetune"
    # tensorboard_path = "tensorboard/dqn_3o_15_bigger_buffer_finetune/"
    # weight_path = "weights_3o_15_bigger_buffer_finetune"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory()

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_M_C_5(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=1000)
    
    finished = True
    for i in range(start_to_train):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        action_index = np.random.randint(action_space)
        action = np.zeros(action_space)
        action[action_index] = 1
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
    
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    for step in range(total_episodes):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space)
            action = np.zeros(action_space)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space)
            action[choice] = 1
        
        action_index = np.argmax(action)
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_M_C_5(size=(15,15))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate()
                Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                while True:
                    action = np.argmax(Qs)
                    reward, done = env_test.move(action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train_two_stage_task():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 200              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 5
    start_to_train = 100000
    state_one_size = [15,15,11]
    state_two_size = [15,15,5]

    # possible_actions = [(1,0,0),(0,1,0),(0,0,1)]

    tensorboard_path = "tensorboard/dqn_new_task/"
    weight_path = "weights_dqn_new_task"
    tensorboard_path = "tensorboard/dqn_new_task_continue/"
    weight_path = "weights_dqn_new_task_continue"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net_stage_one = DQNetwork3(state_size=state_one_size,learning_rate=learning_rate,action_size=action_space,name="stage_one")
    net_stage_two = DQNetwork3(state_size=state_two_size,learning_rate=learning_rate,action_size=action_space,name="stage_two")

    '''
        Setup buffer
    '''
    buffer = Memory()

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss_stage_one", net_stage_one.loss)
    tf.summary.scalar("Loss_stage_two", net_stage_two.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    times = 0
    for i in range(start_to_train):
        if finished or times > max_steps:
            env.randominit()
            finished = False
            times = 0
        
        state_one = env.getstate_1()
        choice_index = np.random.randint(action_space)
        choice_action = np.random.randint(action_space)
        choice_index_next = np.random.randint(action_space)
        action_one = np.zeros(action_space)
        action_one[choice_index] = 1
        action_two = np.zeros(action_space)
        action_two[choice_action] = 1
        state_two = env.getstate_2(choice_index)

        reward, done = env.move(choice_index,choice_action)
        next_state_one = env.getstate_1()
        next_state_two = env.getstate_2(choice_index_next)

        buffer.add((state_one, action_one, state_two, action_two, reward, next_state_one, next_state_two, done))
        finished = done
        times += 1
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    saver.restore(sess,'./weights_dqn_new_task/model_1100000.ckpt')
    decay_step = 0

    finished = True
    times = 0
    for step in range(1100000,total_episodes):
        if finished or times > max_steps:
            env.randominit()
            finished = False
            times = 0
        
        state_one = env.getstate_1()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            choice_index = np.random.randint(action_space)
            choice_action = np.random.randint(action_space)
            action_one = np.zeros(action_space)
            action_one[choice_index] = 1
            action_two = np.zeros(action_space)
            action_two[choice_action] = 1
            state_two = env.getstate_2(choice_action)
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net_stage_one.output, feed_dict = {net_stage_one.inputs_: state_one.reshape((1, *state_one.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action_one = np.zeros(action_space)
            action_one[choice] = 1
            
            
            state_two = env.getstate_2(choice)
            Qs = sess.run(net_stage_two.output, feed_dict = {net_stage_two.inputs_: state_two.reshape((1, *state_two.shape))})
            
            choice = np.argmax(Qs)
            action_two = np.zeros(action_space)
            action_two[choice] = 1
        
        choice_index = np.argmax(action_one)
        choice_action = np.argmax(action_two)
        reward, done = env.move(choice_index,choice_action)
        next_state_one = env.getstate_1()
        Qs = sess.run(net_stage_one.output, feed_dict = {net_stage_one.inputs_: next_state_one.reshape((1, *next_state_one.shape))})
        choice_index_next = np.argmax(Qs)
        next_state_two = env.getstate_2(choice_index_next)

        buffer.add((state_one, action_one, state_two, action_two, reward, next_state_one, next_state_two, done))
        finished = done
        decay_step += 1
        times += 1

        batch = buffer.sample(batch_size)
        states_one_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_one_mb = np.array([each[1] for each in batch])
        states_two_mb = np.array([each[2] for each in batch], ndmin=3)
        actions_two_mb = np.array([each[3] for each in batch])
        rewards_mb = np.array([each[4] for each in batch])
        next_states_one_mb = np.array([each[5] for each in batch])
        next_states_two_mb = np.array([each[6] for each in batch])
        dones_mb = np.array([each[7] for each in batch])

        target_Qs_batch_one = []
        target_Qs_batch_two = []
        
        Qs_next_state_one = sess.run(net_stage_one.output, feed_dict = {net_stage_one.inputs_: next_states_one_mb})
        Qs_next_state_two = sess.run(net_stage_two.output, feed_dict = {net_stage_two.inputs_: next_states_two_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch_one.append(rewards_mb[i])
                target_Qs_batch_two.append(rewards_mb[i])
            else:
                target_one = rewards_mb[i] + gamma * np.max(Qs_next_state_one[i])
                target_Qs_batch_one.append(target_one)
                target_two = rewards_mb[i] + gamma * np.max(Qs_next_state_two[i])
                target_Qs_batch_two.append(target_two)
        
        targets_mb_one = np.array([each for each in target_Qs_batch_one])
        targets_mb_two = np.array([each for each in target_Qs_batch_two])

        _, _, summary = sess.run([net_stage_one.optimizer,net_stage_two.optimizer, write_op],feed_dict={net_stage_one.inputs_ : states_one_mb, net_stage_one.target_Q : targets_mb_one, net_stage_one.actions_: actions_one_mb,
        net_stage_two.inputs_ : states_two_mb, net_stage_two.target_Q : targets_mb_two, net_stage_two.actions_: actions_two_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_scene(size=(15,15))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 50:
                s += 1
                state_one = env_test.getstate_1()
                Qs = sess.run(net_stage_one.output, feed_dict={net_stage_one.inputs_: state_one.reshape((1, *state_one.shape))})
                Qs = Qs.squeeze()
                while True:
                    choice_index = np.argmax(Qs)
                    state_two = env.getstate_2(choice_index)
                    Qs_ = sess.run(net_stage_two.output, feed_dict={net_stage_two.inputs_: state_two.reshape((1, *state_two.shape))})
                    Qs_ = Qs_.squeeze()
                    flag = False
                    cnt = 0

                    while cnt < action_space:
                        cnt += 1
                        choice_action = np.argmax(Qs_)
                        reward, done = env_test.move(choice_index, choice_action)
                        total_reward += reward
                        if done == -1:
                            Qs_[choice_action] = -1000000000
                            continue
                        flag = True
                        break
                    
                    if done == -1:
                        done = 0
                    
                    if not flag:
                        Qs[choice_index] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train_comb():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    obj_num = 5
    action_space = 5
    start_to_train = 100000
    state_size = [15,15,2*obj_num+1]


    tensorboard_path = "tensorboard/dqn_comb_new_task_biglr/"
    weight_path = "weights_comb_new_task_biglr"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(600000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    for i in range(start_to_train):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate_1()
        action_index = np.random.randint(obj_num*action_space)
        action = np.zeros(action_space*obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        buffer.add((state, action, reward, next_state, done))
        finished = done
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    for step in range(total_episodes):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate_1()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_scene(size=(15,15))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate_1()
                Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                while True:
                    action = np.argmax(Qs)
                    choice_index = int(action/action_space)
                    choice_action = action%action_space
                    reward, done = env_test.move(choice_index, choice_action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train_comb_new_action():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0001
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.2            # minimum exploration probability 
    decay_rate = 0.00001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    obj_num = 5
    action_space = 5
    start_to_train = 100000
    state_size = [15, 15, 2*obj_num+1]


    # tensorboard_path = "tensorboard/dqn_comb_new_task_new_action/"
    # weight_path = "weights_comb_new_task_new_action"
    tensorboard_path = "tensorboard/20181011_1_continue/"
    weight_path = "weights_20181011_1_continue"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(300000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    for i in range(start_to_train):
        if finished:
            ra = np.random.rand()
            if ra < 0.8:
                env.randominit_crowded()
            else:
                env.randominit()
            finished = False
        
        state = env.getstate_1()
        action_index = np.random.randint(obj_num*action_space)
        action = np.zeros(action_space*obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        print('task %d'%i)
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    saver.restore(sess,'./weights_20181011_1/model_2040000.ckpt')
    decay_step = 0

    finished = True
    for step in range(2040001,total_episodes):
        if finished:
            ra = np.random.rand()
            if ra < 0.8:
                env.randominit_crowded()
            else:
                env.randominit()
            finished = False
        
        state = env.getstate_1()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = 1.0 # to comment!!
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_scene(size=(15,15))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate_1()
                Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                while True:
                    action = np.argmax(Qs)
                    choice_index = int(action/action_space)
                    choice_action = action%action_space
                    reward, done = env_test.move(choice_index, choice_action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train_comb_two_frame():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 5000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    obj_num = 5
    action_space = 5
    start_to_train = 100000
    state_size = [15,15,(2*obj_num+1)*2]


    tensorboard_path = "tensorboard/dqn_comb_new_task_two_frame/"
    weight_path = "weights_comb_new_task_two_frame"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(250000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    prestate = None
    for i in range(start_to_train):
        if finished:
            env.randominit()
            prestate = env.getstate_1()
            finished = False

        state = env.getstate_1()
        action_index = np.random.randint(obj_num*action_space)
        action = np.zeros(action_space*obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        
        cur_two_frame = np.concatenate([prestate,state],-1)
        nex_two_frame = np.concatenate([state,next_state],-1)
        buffer.add((cur_two_frame, action, reward, nex_two_frame, done))
        finished = done
        prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    prestate = None
    for step in range(total_episodes):
        if finished:
            env.randominit()
            prestate = env.getstate_1()
            finished = False
        
        state = env.getstate_1()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        cur_two_frame = np.concatenate([prestate,state],-1)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        nex_two_frame = np.concatenate([state, next_state],-1)
        prestate = state

        buffer.add((cur_two_frame, action, reward, nex_two_frame, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_scene(size=(15,15))
            env_test.randominit()
            prestate = env_test.getstate_1()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate_1()
                cur_two_frame = np.concatenate([prestate,state],-1)

                Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape))})
                Qs = Qs.squeeze()

                prestate = state
                while True:
                    action = np.argmax(Qs)
                    choice_index = int(action/action_space)
                    choice_action = action%action_space
                    reward, done = env_test.move(choice_index, choice_action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train_comb_five_frame():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 5000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.00001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100000
    state_size = [15,15,(2*obj_num+1)*frame_num]


    tensorboard_path = "tensorboard/20181019_2/"
    weight_path = "weights_20181019_2"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(150000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    for i in range(start_to_train):
        if finished:
            state_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False

        state = env.getstate_1()
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        state_list.append(state)
        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        buffer.add((cur_two_frame, action, reward, nex_two_frame, done))
        finished = done
        print(i,nex_two_frame.shape)
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    for step in range(total_episodes):
        if finished:
            state_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
        
        state = env.getstate_1()

        state_list.append(state)
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
    

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()


        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)


        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state

        buffer.add((cur_two_frame, action, reward, nex_two_frame, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        if step < 20000000:
            _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        else:
            _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})

        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_scene_new_action_pre_state(size=(15,15))
            sum_reward = []
            count = 0
            for k in range(200):
                env_test.randominit_crowded()
                state_list_test = []
                finished_test = False
                total_reward = 0
                s = 0

                while not finished_test and s < 10:
                    s += 1
                    state = env_test.getstate_1()

                    state_list_test.append(state)

                    if len(state_list_test) < frame_num:
                        dif = frame_num - len(state_list_test)
                        tocon = []
                        for j in range(dif):
                            tocon.append(np.zeros_like(state))

                        cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                            
                    else:
                        cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                    Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape))})
                    Qs = Qs.squeeze()

                    while True:
                        action = np.argmax(Qs)
                        choice_index = int(action/action_space)
                        choice_action = action%action_space
                        reward, done = env_test.move(choice_index, choice_action)
                        total_reward += reward
                        if done == -1:
                            Qs[action] = -1000000000
                            continue
                        
                        finished_test = done
                        break

                sum_reward.append(total_reward)
                if finished_test:
                    count += 1
            
            # sum_reward /= 100
            sum_ = np.mean(sum_reward)
            median_ = np.median(sum_reward)
            count /= 200
            summt.value.add(tag='reward_test',simple_value=sum_)
            summt.value.add(tag='reward_test_median',simple_value=median_)
            summt.value.add(tag='success rate',simple_value=count)
            writer.add_summary(summt, step)

        if step % 1000 == 0 and step > 0: # !!!!! have been modified!!
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def train_comb_five_frame_add_action():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 1000              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.00001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 10000
    state_size = [15,15,(2*obj_num+1)*frame_num]


    tensorboard_path = "tensorboard/20181029_1/"
    weight_path = "weights_20181029_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork4(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space, frame_num=frame_num)
    
    '''
        Setup buffer
    '''
    buffer = Memory(75000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = env.getstate_1()
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        state_list.append(state)
        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))
            
            nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
            tocon.append(np.zeros_like(action))
            cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        action_list.append(action)
        buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        finished = done
        print(i,nex_two_frame.shape,nex_action_chain.shape)
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 4990001
    op_step = 0
    mx_steps = 0
    for step in range(started_from,total_episodes):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        state_list.append(state)
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(action))

            cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()


        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))

            nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        else:
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        action_list.append(action)
        buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/2)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[2] for each in batch])
            rewards_mb = np.array([each[3] for each in batch])
            next_states_mb = np.array([each[4] for each in batch])
            next_action_chain_mb = np.array([each[5] for each in batch])
            dones_mb = np.array([each[6] for each in batch])

            target_Qs_batch = []
            Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb, net.action_chain: next_action_chain_mb})
            
            for i in range(batch_size):
                done = dones_mb[i]
                if done == 1:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)
            
            targets_mb = np.array([each for each in target_Qs_batch])

            if step < 20000000:
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state(size=(15,15))
                sum_reward = []
                count = 0
                for k in range(200):
                    env_test.randominit_crowded()
                    state_list_test = []
                    action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 10:
                        s += 1
                        state = env_test.getstate_1()

                        state_list_test.append(state)

                        if len(state_list_test) < frame_num:
                            dif = frame_num - len(state_list_test)
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros_like(state))

                            cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        else:
                            cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        if len(action_list_test) < frame_num-1:
                            dif = frame_num - len(action_list_test) - 1
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros(action_space * obj_num))
                            
                            cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        else:
                            cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action/action_space)
                            choice_action = action%action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        action_list_test.append(action)
                    if finished_test:
                        count += 1
                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= 200
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 10000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_five_frame_add_action_all_reward():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 1000              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 1000
    state_size = [15,15,(2*obj_num+1)*frame_num]
    action_size = obj_num * action_space


    tensorboard_path = "tensorboard/20181102_1/"
    weight_path = "weights_20181102_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork5(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space, frame_num=frame_num)
    
    '''
        Setup buffer
    '''
    buffer = Memory(5000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)


            

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        
        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))
            
            nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
            tocon.append(np.zeros_like(action))
            cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_frame_pack, nex_action_pack, done))
        finished = done
        print(i,nex_frame_pack.shape,nex_action_pack.shape)
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    saver.restore(sess,'./weights_20181029_2/model_281000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    # started_from = 0
    # started_from = 4990001
    started_from = 281000
    op_step = 0
    mx_steps = 0
    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(action))

            cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()


        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))

            nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        else:
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_frame_pack, nex_action_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[2] for each in batch])
            rewards_mb = np.array([each[3] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[4] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            next_action_chain_mb = np.array([each[5] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)
            dones_mb = np.array([each[6] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:,a], net.action_chain: next_action_chain_mb[:,a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else: 
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty(size=(15,15))
                sum_reward = []
                count = 0
                for k in range(200):
                    env_test.randominit_crowded()
                    state_list_test = []
                    action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        state_list_test.append(state)

                        if len(state_list_test) < frame_num:
                            dif = frame_num - len(state_list_test)
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros_like(state))

                            cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        else:
                            cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        if len(action_list_test) < frame_num-1:
                            dif = frame_num - len(action_list_test) - 1
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros(action_space * obj_num))
                            
                            cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        else:
                            cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action/action_space)
                            choice_action = action%action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        action_list_test.append(action)
                    if finished_test:
                        count += 1
                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= 200
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_five_frame_add_action_all_reward_loss_details():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 1000              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100
    state_size = [15,15,(2*obj_num+1)*frame_num]
    action_size = obj_num * action_space
    
    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181105_2/"
    weight_path = "weights_20181105_2"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork6(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space, frame_num=frame_num)
    
    '''
        Setup buffer
    '''
    buffer = Memory(10000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)


            

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        
        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))
            
            nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
            tocon.append(np.zeros_like(action))
            cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_frame_pack, nex_action_pack, done))
        finished = done
        print(i,nex_frame_pack.shape,nex_action_pack.shape)
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    # started_from = 0
    # started_from = 4990001
    # started_from = 281000
    started_from = 341000
    op_step = 0
    mx_steps = 0
    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(action))

            cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()


        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))

            nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        else:
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_frame_pack, nex_action_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[2] for each in batch])
            rewards_mb = np.array([each[3] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[4] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            next_action_chain_mb = np.array([each[5] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)
            dones_mb = np.array([each[6] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:,a], net.action_chain: next_action_chain_mb[:,a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty(size=(15,15))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    state_list_test = []
                    action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        state_list_test.append(state)

                        if len(state_list_test) < frame_num:
                            dif = frame_num - len(state_list_test)
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros_like(state))

                            cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        else:
                            cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        if len(action_list_test) < frame_num-1:
                            dif = frame_num - len(action_list_test) - 1
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros(action_space * obj_num))
                            
                            cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        else:
                            cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        action_list_test.append(action)
                    if finished_test:
                        count += 1
                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_five_frame_add_action_all_reward_loss_details_failure_cases_reinforce():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100
    state_size = [15,15,(2*obj_num+1)*frame_num]
    action_size = obj_num * action_space
    
    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181110_2/"
    weight_path = "weights_20181110_2"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork6(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space, frame_num=frame_num)
    
    '''
        Setup buffer
    '''
    buffer = Memory(10000)
    failure_buffer = Memory(200)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)


            

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        
        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))
            
            nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
            tocon.append(np.zeros_like(action))
            cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_frame_pack, nex_action_pack, done))
        finished = done
        print(i,nex_frame_pack.shape,nex_action_pack.shape)
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
            else:
                env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(action))

            cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()


        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))

            nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        else:
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_frame_pack, nex_action_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[2] for each in batch])
            rewards_mb = np.array([each[3] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[4] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            next_action_chain_mb = np.array([each[5] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)
            dones_mb = np.array([each[6] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:,a], net.action_chain: next_action_chain_mb[:,a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer,feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty(size=(15,15))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    state_list_test = []
                    action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        state_list_test.append(state)

                        if len(state_list_test) < frame_num:
                            dif = frame_num - len(state_list_test)
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros_like(state))

                            cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        else:
                            cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        if len(action_list_test) < frame_num-1:
                            dif = frame_num - len(action_list_test) - 1
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros(action_space * obj_num))
                            
                            cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        else:
                            cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        action_list_test.append(action)

                    if finished_test:
                        count += 1
                    else:
                        this_case = env_test.getconfig()
                        failure_buffer.add(this_case)

                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_five_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100
    state_size = [15,15,(2*obj_num+1)*frame_num]
    action_size = obj_num * action_space
    
    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181110_4/"
    weight_path = "weights_20181110_4"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork7(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space, frame_num=frame_num)
    
    '''
        Setup buffer
    '''
    buffer = Memory(5000)
    failure_buffer = Memory(200)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon,*action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_frame_pack.shape,nex_action_pack.shape,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        
        

        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))
            
            nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
            tocon.append(np.zeros_like(action))
            cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_action_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
            else:
                env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        state_list.append(state)

        rewards = []
        nex_frame_pack = []
        nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []
        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)


            if len(state_list) < frame_num:
                dif = frame_num - len(state_list)
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(state))

                nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                tocon.append(np.zeros_like(state))
                cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            else:
                cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
                nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            if len(action_list) < frame_num-1:
                dif = frame_num - len(action_list) - 1
                tocon = []
                for j in range(dif-1):
                    tocon.append(np.zeros_like(action))
                
                nex_action_chain = np.concatenate([*tocon, *action_list, action],-1)
                tocon.append(np.zeros_like(action))
                cur_action_chain = np.concatenate([*tocon, *action_list],-1)

            else:
                cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
                nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(nex_two_frame)
            nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(action))

            cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        



        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))

            nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        else:
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((cur_two_frame, cur_action_chain, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_action_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[2] for each in batch])
            rewards_mb = np.array([each[3] for each in batch])
            conflict_matrix_mb = np.array([each[4] for each in batch])
            finish_tag_mb = np.array([each[5] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[6] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[8] for each in batch])
            finish_tag_next_mb = np.array([each[9] for each in batch])

            dones_mb = np.array([each[10] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.action_chain: next_action_chain_mb[:,a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict(size=(15,15))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    state_list_test = []
                    action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        state_list_test.append(state)

                        if len(state_list_test) < frame_num:
                            dif = frame_num - len(state_list_test)
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros_like(state))

                            cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        else:
                            cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        if len(action_list_test) < frame_num-1:
                            dif = frame_num - len(action_list_test) - 1
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros(action_space * obj_num))
                            
                            cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        else:
                            cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        conflict_matrix = env_test.getconflict()
                        finish_tag = env_test.getfinished()

                        Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        action_list_test.append(action)

                    if finished_test:
                        count += 1
                    else:
                        this_case = env_test.getconfig()
                        failure_buffer.add(this_case)

                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    # frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100
    state_size = [15,15,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181128_1/"
    weight_path = "weights_20181128_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork8(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(50000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_easy(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            # if len(state_list) < frame_num:
            #     dif = frame_num - len(state_list)
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(state))

            #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            #     tocon.append(np.zeros_like(state))
            #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            # else:
            #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_frame_pack.shape,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        
        

        
        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(state))

        #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
        #     tocon.append(np.zeros_like(state))
        #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        # else:
        #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
        #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(action))
            
        #     nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
        #     tocon.append(np.zeros_like(action))
        #     cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        # else:
        #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
        #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
            else:
                env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)


            # if len(state_list) < frame_num:
            #     dif = frame_num - len(state_list)
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(state))

            #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            #     tocon.append(np.zeros_like(state))
            #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            # else:
            #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            # if len(action_list) < frame_num-1:
            #     dif = frame_num - len(action_list) - 1
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(action))
                
            #     nex_action_chain = np.concatenate([*tocon, *action_list, action],-1)
            #     tocon.append(np.zeros_like(action))
            #     cur_action_chain = np.concatenate([*tocon, *action_list],-1)

            # else:
            #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif):
        #         tocon.append(np.zeros_like(state))

        #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        # else:
        #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif):
        #         tocon.append(np.zeros_like(action))

        #     cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        # else:
        #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        



        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(state))
            
        #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        # else:
        #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(action))

        #     nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        # else:
        #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[5] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_easy(size=(15,15))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    # state_list_test = []
                    # action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        # state_list_test.append(state)

                        # if len(state_list_test) < frame_num:
                        #     dif = frame_num - len(state_list_test)
                        #     tocon = []
                        #     for j in range(dif):
                        #         tocon.append(np.zeros_like(state))

                        #     cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        # else:
                        #     cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        # if len(action_list_test) < frame_num-1:
                        #     dif = frame_num - len(action_list_test) - 1
                        #     tocon = []
                        #     for j in range(dif):
                        #         tocon.append(np.zeros(action_space * obj_num))
                            
                        #     cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        # else:
                        #     cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        conflict_matrix = env_test.getconflict()
                        finish_tag = env_test.getfinished()

                        Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        # action_list_test.append(action)

                    if finished_test:
                        count += 1
                    else:
                        this_case = env_test.getconfig()
                        failure_buffer.add(this_case)

                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 12
    action_space = 5
    start_to_train = 100
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    with open('test35.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181208_2/"
    weight_path = "weights_20181208_2"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork8(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(5000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            env.randominit_crowded(obj_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_frame_pack.shape,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        
        

      

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
            else:
                env.randominit_crowded(obj_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[5] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    # state_list_test = []
                    # action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        conflict_matrix = env_test.getconflict()
                        finish_tag = env_test.getfinished()

                        Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        # action_list_test.append(action)

                    if finished_test:
                        count += 1
                    else:
                        this_case = env_test.getconfig()
                        failure_buffer.add(this_case)

                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1


def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_12():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 12
    action_space = 5
    start_to_train = 100
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [3,5,8,12]
    with open('./test35_3.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    with open('./test35_5.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('./test35_8.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('test35.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20181210_1/"
    weight_path = "weights_20181210_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork8(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(1000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num)+1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()
            next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

            conflict_matrix = env_copy.getconflict()
            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_frame_pack.shape,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

      

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 15001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
                current_num = len(pos_)
            else:
                current_num = np.random.randint(obj_num)+1
                env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()
        state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()
            next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

            conflict_matrix = env_copy.getconflict()
            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[5] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_12_memory_saving():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 12
    action_space = 5
    start_to_train = 100
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [3,5,8,12]
    with open('./test35_3.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    with open('./test35_5.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('./test35_8.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('test35.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20181210_4/"
    weight_path = "weights_20181210_4"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork8(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(100000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

      

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
                current_num = len(pos_)
            else:
                current_num = np.random.randint(obj_num)+1
                env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(map_size,map_size))
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_12_memory_saving_random_index():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 12
    action_space = 5
    start_to_train = 12800
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    # item_num_list = [3,5,8,12]
    # with open('./test35_3.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)

    # with open('./test35_5.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    
    # with open('./test35_8.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    
    # with open('test35.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    item_num_list = [9]
    with open('./test35_9.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    tensorboard_path = "tensorboard/20181218_3/"
    weight_path = "weights_20181218_3"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork8(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(10000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    # saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    # saver.restore(sess,'./weights_20181210_8/model_8000.ckpt')
    # saver.restore(sess,'./weights_20181211_1/model_21200.ckpt')
    decay_step = 0

    finished = True
    started_from = 0
    # prestate = None
    # started_from = 6610001
    # started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    # started_from = 8001
    # started_from = 21201
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            # ratio = np.random.rand()
            # if ratio < failure_sample and failure_buffer.size() > 0:
            #     case = failure_buffer.sample(1)[0]
            #     pos_,target_,size_ = deepcopy(case)
            #     env.setmap(pos_,target_,size_)
            #     current_num = len(pos_)
            # else:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            # state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb = np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 200 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1


def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 17
    action_space = 5
    start_to_train = 1280
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [5,9,13,17]
    with open('./test35_5.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    with open('./test35_9.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('./test35_13.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('test35_17.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20181221_1/"
    weight_path = "weights_20181221_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork8(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(30000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    # saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    # saver.restore(sess,'./weights_20181210_8/model_8000.ckpt')
    # saver.restore(sess,'./weights_20181211_1/model_21200.ckpt')
    saver.restore(sess,'./weights_20181218_1/model_17400.ckpt')
    decay_step = 0

    finished = True
    started_from = 0
    # prestate = None
    # started_from = 6610001
    # started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    # started_from = 8001
    # started_from = 21201
    started_from = 17401
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            # ratio = np.random.rand()
            # if ratio < failure_sample and failure_buffer.size() > 0:
            #     case = failure_buffer.sample(1)[0]
            #     pos_,target_,size_ = deepcopy(case)
            #     env.setmap(pos_,target_,size_)
            #     current_num = len(pos_)
            # else:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            # state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb = np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 200 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index_NN11():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 17
    action_space = 5
    start_to_train = 1280
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [5,9,13,17]
    with open('./test35_5.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    with open('./test35_9.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('./test35_13.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('test35_17.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20181227_2/"
    weight_path = "weights_20181227_2"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork11(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(30000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    # saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    # saver.restore(sess,'./weights_20181210_8/model_8000.ckpt')
    # saver.restore(sess,'./weights_20181211_1/model_21200.ckpt')
    # saver.restore(sess,'./weights_20181218_1/model_17400.ckpt') #zhushi
    decay_step = 0

    finished = True
    started_from = 0
    # prestate = None
    # started_from = 6610001
    # started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    # started_from = 8001
    # started_from = 21201
    #started_from = 17401
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            # ratio = np.random.rand()
            # if ratio < failure_sample and failure_buffer.size() > 0:
            #     case = failure_buffer.sample(1)[0]
            #     pos_,target_,size_ = deepcopy(case)
            #     env.setmap(pos_,target_,size_)
            #     current_num = len(pos_)
            # else:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            # state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb = np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 200 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index_NN12():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 17
    action_space = 5
    start_to_train = 256
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [5,9,13,17]
    with open('./test35_5.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    with open('./test35_9.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('./test35_13.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('test35_17.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20190103_1/"
    weight_path = "weights_20190103_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(30000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    # saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    # saver.restore(sess,'./weights_20181210_8/model_8000.ckpt')
    # saver.restore(sess,'./weights_20181211_1/model_21200.ckpt')
    # saver.restore(sess,'./weights_20181218_1/model_17400.ckpt') #zhushi
    decay_step = 0

    finished = True
    started_from = 0
    # prestate = None
    # started_from = 6610001
    # started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    # started_from = 8001
    # started_from = 21201
    #started_from = 17401
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            # ratio = np.random.rand()
            # if ratio < failure_sample and failure_buffer.size() > 0:
            #     case = failure_buffer.sample(1)[0]
            #     pos_,target_,size_ = deepcopy(case)
            #     env.setmap(pos_,target_,size_)
            #     current_num = len(pos_)
            # else:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            # state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb = np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 200 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1


def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_25_memory_saving_random_index_NN12():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 25
    action_space = 5
    start_to_train = 256
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [5,9,13,17]

    # with open('./test35_5.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./test35_9.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./test35_13.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('test35_17.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20190109_1/"
    weight_path = "weights_20190109_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(10000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    # saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    # saver.restore(sess,'./weights_20181210_8/model_8000.ckpt')
    # saver.restore(sess,'./weights_20181211_1/model_21200.ckpt')
    # saver.restore(sess,'./weights_20181218_1/model_17400.ckpt') #zhushi
    decay_step = 0

    finished = True
    started_from = 0
    # prestate = None
    # started_from = 6610001
    # started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    # started_from = 8001
    # started_from = 21201
    #started_from = 17401
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            # ratio = np.random.rand()
            # if ratio < failure_sample and failure_buffer.size() > 0:
            #     case = failure_buffer.sample(1)[0]
            #     pos_,target_,size_ = deepcopy(case)
            #     env.setmap(pos_,target_,size_)
            #     current_num = len(pos_)
            # else:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            # state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb = np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 200 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1



def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index_NN13():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 17
    action_space = 5
    start_to_train = 256
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [5,9,13,17]
    with open('./test35_5.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    with open('./test35_9.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('./test35_13.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)
    
    with open('test35_17.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)


    tensorboard_path = "tensorboard/20190104_1/"
    weight_path = "weights_20190104_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork13(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(3000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    current_num = obj_num
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        
        

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    # saver.restore(sess,'./weights_20181209_2/model_15000.ckpt')
    # saver.restore(sess,'./weights_20181210_1/model_21000.ckpt')
    # saver.restore(sess,'./weights_20181210_8/model_8000.ckpt')
    # saver.restore(sess,'./weights_20181211_1/model_21200.ckpt')
    # saver.restore(sess,'./weights_20181218_1/model_17400.ckpt') #zhushi
    decay_step = 0

    finished = True
    started_from = 0
    # prestate = None
    # started_from = 6610001
    # started_from = 21001
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    # started_from = 8001
    # started_from = 21201
    #started_from = 17401
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            # ratio = np.random.rand()
            # if ratio < failure_sample and failure_buffer.size() > 0:
            #     case = failure_buffer.sample(1)[0]
            #     pos_,target_,size_ = deepcopy(case)
            #     env.setmap(pos_,target_,size_)
            #     current_num = len(pos_)
            # else:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = deepcopy(env.getconfig())
        # state = env.getstate_1()
        # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            # next_state = env_copy.getstate_1()
            # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
            next_state = deepcopy(env_copy.getconfig())

            conflict_matrix = env_copy.getconflict()
            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
            finish_tag = env_copy.getfinished()
            # finish_tag = np.pad(finish_tag,((0,obj_num-current_num)),"constant")

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        # nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        conflict_matrix = env.getconflict()
        # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num),(0,obj_num-current_num),(0,0)),"constant")
        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            state_ = env.getstate_1()
            # state_ = np.pad(state_,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")
        # state_list.append(state)
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state_.reshape((1, *state_.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        # next_state = env.getstate_1()
        # next_state = np.pad(next_state,((0,0),(0,0),(0,(obj_num-current_num)*2)),"constant")

        
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # states_mb = np.array([each[0] for each in batch], ndmin=3)
            configs_mb = [each[0] for each in batch]
            states_mb = []
            env_made = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
            
            for state_config in configs_mb:
                pos,target,size = state_config
                sub_num = obj_num - len(pos)
                for _ in range(sub_num):
                    pos.append((0,0))
                    target.append((0,0))
                    size.append((0,0))

                env_made.setmap(pos,target,size)
                states_mb.append(env_made.getstate_1())
            
            states_mb = np.array(states_mb)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            # next_states_mb = np.array([each[5] for each in batch])
            configs_next_mb = [each[5] for each in batch]
            next_states_mb = []

            for configs in configs_next_mb:
                next_states_action = []
                for config in configs:
                    pos, target, size = config
                    sub_num = obj_num - len(pos)
                    for _ in range(sub_num):
                        pos.append((0,0))
                        target.append((0,0))
                        size.append((0,0))
                    
                    env_made.setmap(pos,target,size)
                    next_states_action.append(env_made.getstate_1())
                
                next_states_mb.append(next_states_action)
                
            next_states_mb = np.array(next_states_mb)

            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb = np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
                

                for ii,test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for test_config in test_configs:
                        pos_,target_,size_ = deepcopy(test_config)
                        env_test.setmap(pos_,target_,size_)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0
                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_1()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            conflict_matrix = env_test.getconflict()
                            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    Qs[action] = -1000000000
                                    continue
                                
                                finished_test = done
                                break

                            action_index = action
                            action = np.zeros(action_space * obj_num)
                            action[action_index] = 1
                            # action_list_test.append(action)

                        if finished_test:
                            count += 1
                        else:
                            this_case = env_test.getconfig()
                            failure_buffer.add(this_case)

                        sum_reward.append(total_reward)
                    
                    # sum_reward /= 100
                    sum_ = np.mean(sum_reward)
                    median_ = np.median(sum_reward)
                    count /= len(test_configs)
                    summt.value.add(tag='reward_test_%d'%(current_num_test),simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d'%(current_num_test),simple_value=median_)
                    summt.value.add(tag='success rate_%d'%(current_num_test),simple_value=count)
                    writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 200 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1


def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_no_conflict():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    # frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100
    state_size = [15,15,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181128_2/"
    weight_path = "weights_20181128_2"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork9(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(50000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_easy(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            # if len(state_list) < frame_num:
            #     dif = frame_num - len(state_list)
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(state))

            #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            #     tocon.append(np.zeros_like(state))
            #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            # else:
            #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_frame_pack.shape,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        
        

        
        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(state))

        #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
        #     tocon.append(np.zeros_like(state))
        #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        # else:
        #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
        #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(action))
            
        #     nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
        #     tocon.append(np.zeros_like(action))
        #     cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        # else:
        #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
        #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
            else:
                env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)


            # if len(state_list) < frame_num:
            #     dif = frame_num - len(state_list)
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(state))

            #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            #     tocon.append(np.zeros_like(state))
            #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            # else:
            #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            # if len(action_list) < frame_num-1:
            #     dif = frame_num - len(action_list) - 1
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(action))
                
            #     nex_action_chain = np.concatenate([*tocon, *action_list, action],-1)
            #     tocon.append(np.zeros_like(action))
            #     cur_action_chain = np.concatenate([*tocon, *action_list],-1)

            # else:
            #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif):
        #         tocon.append(np.zeros_like(state))

        #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        # else:
        #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif):
        #         tocon.append(np.zeros_like(action))

        #     cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        # else:
        #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        



        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(state))
            
        #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        # else:
        #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(action))

        #     nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        # else:
        #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[5] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.finish_tag: finish_tag_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.finish_tag: finish_tag_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.finish_tag: finish_tag_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.finish_tag: finish_tag_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_easy(size=(15,15))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    # state_list_test = []
                    # action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        # state_list_test.append(state)

                        # if len(state_list_test) < frame_num:
                        #     dif = frame_num - len(state_list_test)
                        #     tocon = []
                        #     for j in range(dif):
                        #         tocon.append(np.zeros_like(state))

                        #     cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        # else:
                        #     cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        # if len(action_list_test) < frame_num-1:
                        #     dif = frame_num - len(action_list_test) - 1
                        #     tocon = []
                        #     for j in range(dif):
                        #         tocon.append(np.zeros(action_space * obj_num))
                            
                        #     cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        # else:
                        #     cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        conflict_matrix = env_test.getconflict()
                        finish_tag = env_test.getfinished()

                        Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        # action_list_test.append(action)

                    if finished_test:
                        count += 1
                    else:
                        this_case = env_test.getconfig()
                        failure_buffer.add(this_case)

                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_no_finish_tag():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    # frame_num = 5
    obj_num = 5
    action_space = 5
    start_to_train = 100
    state_size = [15,15,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    tensorboard_path = "tensorboard/20181128_3/"
    weight_path = "weights_20181128_3"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork10(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory(50000)
    failure_buffer = Memory(2000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_easy(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:"%(i+1),net.loss_details[i])

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    # state_list = []
    # action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        
        state = env.getstate_1()
        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)

            # if len(state_list) < frame_num:
            #     dif = frame_num - len(state_list)
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(state))

            #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            #     tocon.append(np.zeros_like(state))
            #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            # else:
            #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)
        
        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        print(i,nex_frame_pack.shape,nex_conflict_matrix_pack.shape,nex_finish_tag_pack.shape)


        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()
        
        

        
        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(state))

        #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
        #     tocon.append(np.zeros_like(state))
        #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        # else:
        #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
        #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(action))
            
        #     nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
        #     tocon.append(np.zeros_like(action))
        #     cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        # else:
        #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
        #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, rewards, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')
    # saver.restore(sess,'./weights_20181105_3/model_30000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 30000
    # started_from = 4990001
    # started_from = 281000
    # started_from = 341000
    op_step = 0
    mx_steps = 0
    failure_sample = 0.4

    for step in range(started_from, total_episodes):
        if finished or mx_steps == max_steps:
            # state_list = []
            # action_list = []
            ratio = np.random.rand()
            if ratio < failure_sample and failure_buffer.size() > 0:
                case = failure_buffer.sample(1)[0]
                pos_,target_,size_ = deepcopy(case)
                env.setmap(pos_,target_,size_)
            else:
                env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        # state_list.append(state)

        rewards = []
        nex_frame_pack = []
        # nex_action_pack = []
        nex_conflict_matrix_pack = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a/action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_1()

            conflict_matrix = env_copy.getconflict()
            finish_tag = env_copy.getfinished()

            nex_conflict_matrix_pack.append(conflict_matrix)
            nex_finish_tag_pack.append(finish_tag)


            # if len(state_list) < frame_num:
            #     dif = frame_num - len(state_list)
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(state))

            #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            #     tocon.append(np.zeros_like(state))
            #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
            # else:
            #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

            # if len(action_list) < frame_num-1:
            #     dif = frame_num - len(action_list) - 1
            #     tocon = []
            #     for j in range(dif-1):
            #         tocon.append(np.zeros_like(action))
                
            #     nex_action_chain = np.concatenate([*tocon, *action_list, action],-1)
            #     tocon.append(np.zeros_like(action))
            #     cur_action_chain = np.concatenate([*tocon, *action_list],-1)

            # else:
            #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
            
            nex_frame_pack.append(next_state)
            # nex_action_pack.append(nex_action_chain)

        nex_frame_pack = np.array(nex_frame_pack)
        # nex_action_pack = np.array(nex_action_pack)
        nex_conflict_matrix_pack = np.array(nex_conflict_matrix_pack)
        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        # explore_probability = 1
        

        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif):
        #         tocon.append(np.zeros_like(state))

        #     cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        # else:
        #     cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif):
        #         tocon.append(np.zeros_like(action))

        #     cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        # else:
        #     cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        conflict_matrix = env.getconflict()
        finish_tag = env.getfinished()

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        

        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        



        # if len(state_list) < frame_num:
        #     dif = frame_num - len(state_list)
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(state))
            
        #     nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        # else:
        #     nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        # if len(action_list) < frame_num-1:
        #     dif = frame_num - len(action_list) - 1
        #     tocon = []
        #     for j in range(dif-1):
        #         tocon.append(np.zeros_like(action))

        #     nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        # else:
        #     nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        # action_list.append(action)
        # buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        buffer.add((state, action, rewards, conflict_matrix, finish_tag, nex_frame_pack, nex_conflict_matrix_pack, nex_finish_tag_pack, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/4)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            
            # action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            conflict_matrix_mb = np.array([each[3] for each in batch])
            finish_tag_mb = np.array([each[4] for each in batch])
            # print('rewards_mb',rewards_mb.shape)
            next_states_mb = np.array([each[5] for each in batch])
            # print('next_states_mb',next_states_mb.shape)
            # next_action_chain_mb = np.array([each[7] for each in batch])
            # print('next_action_chain_mb',next_action_chain_mb.shape)

            conflict_matrix_next_mb =  np.array([each[6] for each in batch])
            finish_tag_next_mb = np.array([each[7] for each in batch])

            dones_mb = np.array([each[8] for each in batch])
            
            actions_mb = np.ones_like(actions_mb)
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []
            for a in range(action_size):
                Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb[:, a], net.conflict_matrix: conflict_matrix_next_mb[:, a]})
                target_Qs = []
                for i in range(batch_size):
                    done = dones_mb[i]
                    if done == 1:
                        # target_Qs_batch.append(rewards_mb[i])
                        target_Qs.append(rewards_mb[i,a])
                    else:
                        target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
                        # target_Qs_batch.append(target)
                        target_Qs.append(target)
                
                target_Qs_batch.append(target_Qs)
            
            targets_mb = np.array([each for each in target_Qs_batch]).transpose()

            if step < 20000000:
                # optimize two times
                sess.run(net.optimizer, feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb})
                
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.conflict_matrix: conflict_matrix_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 100 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_easy(size=(15,15))
                sum_reward = []
                count = 0
                for test_config in test_configs:
                    pos_,target_,size_ = deepcopy(test_config)
                    env_test.setmap(pos_,target_,size_)
                    # state_list_test = []
                    # action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 20:
                        s += 1
                        state = env_test.getstate_1()

                        # state_list_test.append(state)

                        # if len(state_list_test) < frame_num:
                        #     dif = frame_num - len(state_list_test)
                        #     tocon = []
                        #     for j in range(dif):
                        #         tocon.append(np.zeros_like(state))

                        #     cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        # else:
                        #     cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        # if len(action_list_test) < frame_num-1:
                        #     dif = frame_num - len(action_list_test) - 1
                        #     tocon = []
                        #     for j in range(dif):
                        #         tocon.append(np.zeros(action_space * obj_num))
                            
                        #     cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        # else:
                        #     cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        conflict_matrix = env_test.getconflict()
                        finish_tag = env_test.getfinished()

                        Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action / action_space)
                            choice_action = action % action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        # action_list_test.append(action)

                    if finished_test:
                        count += 1
                    else:
                        this_case = env_test.getconfig()
                        failure_buffer.add(this_case)

                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= len(test_configs)
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_comb_ten_frame_add_action():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 1000              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.00001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    frame_num = 10
    obj_num = 5
    action_space = 5
    start_to_train = 10000
    state_size = [15,15,(2*obj_num+1)*frame_num]


    tensorboard_path = "tensorboard/20181030_1/"
    weight_path = "weights_20181030_1"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetwork4(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space, frame_num=frame_num)
    
    '''
        Setup buffer
    '''
    buffer = Memory(75000)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state(size=(15,15))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    # prestate = None
    state_list = []
    action_list = []
    mx_steps = 0
    for i in range(start_to_train):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0

        state = env.getstate_1()
        action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1
        choice_index = int(action_index/action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()

        state_list.append(state)
        
        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))

            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
            tocon.append(np.zeros_like(state))
            cur_two_frame = np.concatenate([*tocon,*state_list], -1)
                
        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)
        
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))
            
            nex_action_chain = np.concatenate([*tocon,*action_list, action],-1)
            tocon.append(np.zeros_like(action))
            cur_action_chain = np.concatenate([*tocon,*action_list],-1)

        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):],action],-1)
                

        action_list.append(action)
        buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        finished = done
        print(i,nex_two_frame.shape,nex_action_chain.shape)
        mx_steps += 1
        # prestate = state
    
    # sess = tf.InteractiveSession()
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    # saver.restore(sess,'./weights_20181016_1/model_4990000.ckpt')
    # saver.restore(sess,'./weights_20181019_1/model_5220000.ckpt')
    # saver.restore(sess,'./weights_20181020_1/model_6610000.ckpt')
    decay_step = 0

    finished = True
    # prestate = None
    # started_from = 6610001
    started_from = 0
    # started_from = 4990001
    op_step = 0
    mx_steps = 0
    for step in range(started_from,total_episodes):
        if finished or mx_steps == max_steps:
            state_list = []
            action_list = []
            env.randominit_crowded()
            # prestate = env.getstate_1()
            finished = False
            mx_steps = 0
        
        state = env.getstate_1()

        state_list.append(state)
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        # explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = 1
        

        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(state))

            cur_two_frame = np.concatenate([*tocon,*state_list], -1)

        else:
            cur_two_frame = np.concatenate(state_list[-frame_num:],-1)
            
        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif):
                tocon.append(np.zeros_like(action))

            cur_action_chain = np.concatenate([*tocon,*action_list],-1)
        else:
            cur_action_chain = np.concatenate(action_list[-(frame_num-1):],-1)

        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space*obj_num)
            action = np.zeros(action_space*obj_num)
            action[action_index] = 1
            
        else:

            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space*obj_num)
            action[choice] = 1
        
        action_index = np.argmax(action)
        choice_index = int(action_index/action_space)
        choice_action = action_index%action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_1()


        if len(state_list) < frame_num:
            dif = frame_num - len(state_list)
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(state))
            
            nex_two_frame = np.concatenate([*tocon,*state_list,next_state],-1)
                
        else:
            nex_two_frame = np.concatenate([*state_list[-(frame_num-1):],next_state],-1)

        if len(action_list) < frame_num-1:
            dif = frame_num - len(action_list) - 1
            tocon = []
            for j in range(dif-1):
                tocon.append(np.zeros_like(action))

            nex_action_chain = np.concatenate([*tocon,*action_list,action],-1)
        else:
            nex_action_chain = np.concatenate([*action_list[-(frame_num-2):], action],-1)

        # nex_two_frame = np.concatenate([state, next_state],-1)
        # prestate = state
        
        action_list.append(action)
        buffer.add((cur_two_frame, cur_action_chain, action, reward, nex_two_frame, nex_action_chain, done))
        finished = done
        
        
        optimize_frequency = int(batch_size/2)

        if step % optimize_frequency == 0: # prevent the over sampling.
            decay_step += 1
            print(started_from+int((step-started_from)/optimize_frequency),step)
            batch = buffer.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            action_chain_mb = np.array([each[1] for each in batch])
            actions_mb = np.array([each[2] for each in batch])
            rewards_mb = np.array([each[3] for each in batch])
            next_states_mb = np.array([each[4] for each in batch])
            next_action_chain_mb = np.array([each[5] for each in batch])
            dones_mb = np.array([each[6] for each in batch])

            target_Qs_batch = []
            Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb, net.action_chain: next_action_chain_mb})
            
            for i in range(batch_size):
                done = dones_mb[i]
                if done == 1:
                    target_Qs_batch.append(rewards_mb[i])
                else:
                    target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)
            
            targets_mb = np.array([each for each in target_Qs_batch])

            if step < 20000000:
                _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})
            else:
                _, summary = sess.run([net.optimizer2, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb, net.action_chain : action_chain_mb})

            writer.add_summary(summary,started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 1000 == 0 and step > 0:
                summt = tf.Summary()
                env_test = ENV_scene_new_action_pre_state(size=(15,15))
                sum_reward = []
                count = 0
                for k in range(200):
                    env_test.randominit_crowded()
                    state_list_test = []
                    action_list_test = []
                    finished_test = False
                    total_reward = 0
                    s = 0
                    while not finished_test and s < 30:
                        s += 1
                        state = env_test.getstate_1()

                        state_list_test.append(state)

                        if len(state_list_test) < frame_num:
                            dif = frame_num - len(state_list_test)
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros_like(state))

                            cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                                
                        else:
                            cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                        if len(action_list_test) < frame_num-1:
                            dif = frame_num - len(action_list_test) - 1
                            tocon = []
                            for j in range(dif):
                                tocon.append(np.zeros(action_space * obj_num))
                            
                            cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                        else:
                            cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                        Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            choice_index = int(action/action_space)
                            choice_action = action%action_space
                            reward, done = env_test.move(choice_index, choice_action)
                            total_reward += reward
                            if done == -1:
                                Qs[action] = -1000000000
                                continue
                            
                            finished_test = done
                            break

                        action_index = action
                        action = np.zeros(action_space * obj_num)
                        action[action_index] = 1
                        action_list_test.append(action)
                    if finished_test:
                        count += 1
                    sum_reward.append(total_reward)
                
                # sum_reward /= 100
                sum_ = np.mean(sum_reward)
                median_ = np.median(sum_reward)
                count /= 200
                summt.value.add(tag='reward_test',simple_value=sum_)
                summt.value.add(tag='reward_test_median',simple_value=median_)
                summt.value.add(tag='success rate',simple_value=count)
                writer.add_summary(summt, started_from+int((step-started_from)/optimize_frequency))

            if (started_from+int((step-started_from)/optimize_frequency)) % 10000 == 0 and step > 0: # !!!!! have been modified!!
                print('model %d saved'%(started_from+int((step-started_from)/optimize_frequency)))
                saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%(started_from+int((step-started_from)/optimize_frequency))))

        mx_steps += 1

def train_CNN():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.00005
    total_episodes = 50000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.01            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    
    action_space = 30
    start_to_train = 1000
    state_size = [128,128,31]


    tensorboard_path = "tensorboard/dqn_cnn/"
    weight_path = "weights_cnn"


    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = DQNetworkC(state_size=state_size,learning_rate=learning_rate,action_size=action_space)
    
    '''
        Setup buffer
    '''
    buffer = Memory()

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_M_C_L(size=(128,128))

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net.loss)

    write_op = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=100)
    
    finished = True
    for i in range(start_to_train):
        t = time.time()
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        action_index = np.random.randint(action_space)
        action = np.zeros(action_space)
        action[action_index] = 1
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        print(time.time()-t)
    
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_1380000.ckpt')
    # saver.restore(sess,'weights_3o_15/model_306000.ckpt')
    # saver.restore(sess,'weights_3o_15_bigger_buffer/model_510000.ckpt')
    decay_step = 0

    finished = True
    for step in range(total_episodes):
        if finished:
            env.randominit()
            finished = False
        
        state = env.getstate()
        
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            # action = np.random.choice(possible_actions)
            action_index = np.random.randint(action_space)
            action = np.zeros(action_space)
            action[action_index] = 1
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            # action = possible_actions[int(choice)]
            action = np.zeros(action_space)
            action[choice] = 1
        
        action_index = np.argmax(action)
        reward, done = env.move(action_index)
        next_state = env.getstate()
        buffer.add((state, action, reward, next_state, done))
        finished = done
        decay_step += 1

        batch = buffer.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch])
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []
        Qs_next_state = sess.run(net.output, feed_dict = {net.inputs_: next_states_mb})
        
        for i in range(batch_size):
            done = dones_mb[i]
            if done == 1:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        
        targets_mb = np.array([each for each in target_Qs_batch])

        _, summary = sess.run([net.optimizer, write_op],feed_dict={net.inputs_ : states_mb, net.target_Q : targets_mb, net.actions_: actions_mb})
        
        writer.add_summary(summary,step)

        if step % 1000 == 0 and step > 0:
            summt = tf.Summary()
            env_test = ENV_M_C_L(size=(128,128))
            env_test.randominit()
            finished_test = False
            total_reward = 0
            s = 0
            while not finished_test and s < 10:
                s += 1
                state = env_test.getstate()
                Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                while True:
                    action = np.argmax(Qs)
                    reward, done = env_test.move(action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

            summt.value.add(tag='reward_test',simple_value=total_reward)
            writer.add_summary(summt, step)

        if step % 10000 == 0 and step > 0:
            print('model %d saved'%step)
            saver.save(sess,os.path.join(weight_path,'model_%d.ckpt'%step))

def test():
    from visulization import convert
    from PIL import Image
    save_path = 'test'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 3

    env = ENV()
    net = DQNetwork(learning_rate=learning_rate,action_size=action_space)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        saver.restore(sess,'weights/model_498800.ckpt')
        with open('rewards.txt','w') as fp:
            for i in range(100):
                finished = False
                env.randominit()
                step = 0
                init_state = env.getstate()
                img = convert(init_state)
                img = Image.fromarray(np.uint8(img))
                img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                total_reward = 0
                while not finished and step < 10:
                    step += 1
                    state = env.getstate()
                    Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                    action = np.argmax(Qs)
                    reward, done = env.move(action)
                    total_reward += reward
                    state = env.getstate()
                    img = convert(state)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    
                    finished = done

                fp.write('%d\n'%total_reward)
            
def test3():
    from visulization import convert
    from PIL import Image
    save_path = 'test3'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 3
    state_size=[15,15]
    env = ENV3(size=(15,15))
    net = DQNetwork2(state_size=state_size, learning_rate=learning_rate, action_size=action_space)
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    saver.restore(sess,'weights_3o_15_bigger_buffer_v2/model_590000.ckpt')
    with open('rewards3.txt','w') as fp:
        for i in range(100):
            finished = False
            env.randominit()
            step = 0
            init_state = env.getstate()
            img = convert(init_state)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            total_reward = 0
            while not finished and step < 5:
                step += 1
                state = env.getstate()
                Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                while True:
                    action = np.argmax(Qs)
                    Qs = Qs.squeeze()
                    reward, done = env.move(action)
                    total_reward += reward
                    state = env.getstate()
                    img = convert(state)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs[action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_c3():
    from visulization import convert
    from PIL import Image
    save_path = 'test_c3'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 3
    state_size=[15,15,4]
    env = ENV_M_C(size=(15,15))
    net = DQNetwork3(state_size=state_size, learning_rate=learning_rate, action_size=action_space)
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    saver.restore(sess,'weights_3o_15_bigger_buffer_v3/model_1840000.ckpt')

    with open('rewards_c3.txt','w') as fp:
        for i in range(100):
            finished = False
            env.randominit()
            step = 0
            init_state = env.getstate()
            img = env.getmap()
            img = convert(img)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            total_reward = 0
            while not finished and step < 5:
                step += 1
                state = env.getstate()
                Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                while True:
                    action = np.argmax(Qs)
                    Qs = Qs.squeeze()
                    reward, done = env.move(action)
                    total_reward += reward
                    state = env.getstate()
                    mm = env.getmap()
                    img = convert(mm)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs[action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_c5():
    from visulization import convert
    from PIL import Image
    save_path = 'test_c5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 5
    state_size=[15,15,6]
    env = ENV_M_C_5(size=(15,15))
    net = DQNetwork3(state_size=state_size, learning_rate=learning_rate, action_size=action_space)
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    saver.restore(sess,'weights_5o_15_bigger_buffer_v3/model_1000000.ckpt')

    with open('rewards_c5.txt','w') as fp:
        for i in range(100):
            finished = False
            env.randominit()
            step = 0
            init_state = env.getstate()
            img = env.getmap()
            img = convert(img)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            total_reward = 0
            while not finished and step < 5:
                step += 1
                state = env.getstate()
                Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                while True:
                    action = np.argmax(Qs)
                    Qs = Qs.squeeze()
                    reward, done = env.move(action)
                    total_reward += reward
                    state = env.getstate()
                    mm = env.getmap()
                    img = convert(mm)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs[action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_scene():
    from visulization import convert
    from PIL import Image
    save_path = 'test_cene'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 5
    state_one_size = [15,15,11]
    state_two_size = [15,15,5]
    env = ENV_scene(size=(15,15))
    net_stage_one = DQNetwork3(state_size=state_one_size, learning_rate=learning_rate, action_size=action_space, name="stage_one")
    net_stage_two = DQNetwork3(state_size=state_two_size, learning_rate=learning_rate, action_size=action_space, name="stage_two")

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    # saver.restore(sess,'weights_5o_15_bigger_buffer_v3/model_1000000.ckpt')
    saver.restore(sess,'./weights_dqn_new_task/model_640000.ckpt')

    with open('rewards_cene.txt','w') as fp:
        for i in range(100):
            finished = False
            env.randominit()
            step = 0
            img = env.getmap()
            img = convert(img)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            img = env.gettargetmap()
            img = convert(img)
            img = Image.fromarray(img)
            img.save(os.path.join(save_path,'exp_%d_ideal.png'%i))

            total_reward = 0
            while not finished and step < 50:
                step += 1
                state_one = env.getstate_1()
                Qs = sess.run(net_stage_one.output, feed_dict = {net_stage_one.inputs_: state_one.reshape((1, *state_one.shape))})
                choice_index = np.argmax(Qs)
                
                state_two = env.getstate_2(choice_index)
                Qs_ = sess.run(net_stage_two.output, feed_dict = {net_stage_two.inputs_: state_two.reshape((1, *state_two.shape))})
                choice_action = np.argmax(Qs_)
                Qs_ = Qs_.squeeze()
                cnt = 0
                while cnt < action_space:
                    cnt += 1
                    reward, done = env.move(choice_index, choice_action)
                    total_reward += reward
                    # state = env.getstate()
                    mm = env.getmap()
                    img = convert(mm)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs_[choice_action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_comb():
    from visulization import convert
    from PIL import Image
    save_path = 'test_comb_continue_1380000'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 5

    obj_num = 5
    action_space = 5
    state_size = [15,15,2*obj_num+1]
    env = ENV_scene(size=(15,15))
    
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    # saver.restore(sess,'weights_5o_15_bigger_buffer_v3/model_1000000.ckpt')
    # saver.restore(sess,'./weights_dqn_new_task/model_640000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task_continue/model_1790000.ckpt')
    saver.restore(sess,'./weights_comb_new_task_continue/model_1380000.ckpt')

    with open('rewards_comb_continue_1380000.txt','w') as fp:
        for i in range(20):
            finished = False
            env.randominit()
            step = 0
            img = env.getmap()
            img = convert(img)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            img = env.gettargetmap()
            img = convert(img)
            img = Image.fromarray(img)
            img.save(os.path.join(save_path,'exp_%d_ideal.png'%i))

            total_reward = 0
            while not finished and step < 50:
                step += 1
                state = env.getstate_1()
                Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                
                cnt = 0
                while cnt < action_space*obj_num:
                    action = np.argmax(Qs)
                    cnt += 1
                    choice_index = int(action/obj_num)
                    choice_action = action%obj_num

                    reward, done = env.move(choice_index, choice_action)
                    total_reward += reward
                    # state = env.getstate()
                    mm = env.getmap()
                    img = convert(mm)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs[action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_new_action():
    from visulization import convert
    from PIL import Image
    save_path = 'test_comb_new_action'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 5

    obj_num = 5
    action_space = 5
    state_size = [15,15,2*obj_num+1]
    env = ENV_scene_new_action(size=(15,15))
    
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    # saver.restore(sess,'weights_5o_15_bigger_buffer_v3/model_1000000.ckpt')
    # saver.restore(sess,'./weights_dqn_new_task/model_640000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task_continue/model_1790000.ckpt')
    saver.restore(sess,'./weights_comb_new_task_new_action(copy)/model_1070000.ckpt')

    with open('rewards_comb_new_action.txt','w') as fp:
        for i in range(20):
            finished = False
            env.randominit_crowded()
            step = 0
            img = env.getmap()
            img = convert(img)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            img = env.gettargetmap()
            img = convert(img)
            img = Image.fromarray(img)
            img.save(os.path.join(save_path,'exp_%d_ideal.png'%i))

            total_reward = 0
            while not finished and step < 50:
                step += 1
                state = env.getstate_1()
                Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                
                cnt = 0
                while cnt < action_space*obj_num:
                    action = np.argmax(Qs)
                    cnt += 1
                    choice_index = int(action/obj_num)
                    choice_action = action%obj_num

                    reward, done = env.move(choice_index, choice_action)
                    total_reward += reward
                    # state = env.getstate()
                    mm = env.getmap()
                    img = convert(mm)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs[action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_new_action_5_frames():
    from visulization import convert
    from PIL import Image
    save_path = 'test_comb_new_action_5_frames_14000'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 5
    frame_num = 5

    obj_num = 5
    action_space = 5
    # state_size = [15,15,2*obj_num+1]
    state_size = [15,15,(2*obj_num+1)*frame_num]
    env = ENV_scene_new_action(size=(15,15))
    
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    # saver.restore(sess,'weights_5o_15_bigger_buffer_v3/model_1000000.ckpt')
    # saver.restore(sess,'./weights_dqn_new_task/model_640000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task_continue/model_1790000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task_new_action(copy)/model_1070000.ckpt')
    saver.restore(sess,'./weights_20181015_2/model_14000.ckpt')
    # saver.restore(sess,'./weights_20181015_2/model_23000.ckpt')

    with open('rewards_comb_new_action.txt','w') as fp:
        for i in range(20):
            finished = False
            state_list = []
            env.randominit_crowded()
            step = 0
            img = env.getmap()
            img = convert(img)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
            img = env.gettargetmap()
            img = convert(img)
            img = Image.fromarray(img)
            img.save(os.path.join(save_path,'exp_%d_ideal.png'%i))

            total_reward = 0
            while not finished and step < 20:
                step += 1
                cur_state = env.getstate_1()
                state_list.append(cur_state)

                if len(state_list) < frame_num:
                    dif = frame_num - len(state_list)
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros_like(cur_state))
                    
                    state = np.concatenate([*tocon,*state_list], -1)
                        
                else:
                    state = np.concatenate(state_list[-frame_num:],-1)

                Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                Qs = Qs.squeeze()
                
                cnt = 0
                while cnt < action_space*obj_num:
                    action = np.argmax(Qs)
                    cnt += 1
                    choice_index = int(action/obj_num)
                    choice_action = action%obj_num

                    reward, done = env.move(choice_index, choice_action)
                    total_reward += reward
                    # state = env.getstate()
                    mm = env.getmap()
                    img = convert(mm)
                    img = Image.fromarray(np.uint8(img))
                    img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                    if done == -1:
                        Qs[action] = -100000000
                        continue
                    
                    finished = done
                    break
                

            fp.write('%d %d\n'%(i,total_reward))

def test_new_action_5_frames_success_rate():
    from visulization import convert
    from PIL import Image
    save_path = 'test_comb_new_action_5_frames_14000'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    learning_rate = 0.00002
    action_space = 5
    frame_num = 5

    obj_num = 5
    action_space = 5
    # state_size = [15,15,2*obj_num+1]
    state_size = [15,15,(2*obj_num+1)*frame_num]
    env = ENV_scene_new_action(size=(15,15))
    
    net = DQNetwork3(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space)

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    # with tf.InteractiveSession() as sess:
        
    # saver.restore(sess,'weights_3o_15_finetune/model_4990000.ckpt')
    # saver.restore(sess,'weights_5o_15_bigger_buffer_v3/model_1000000.ckpt')
    # saver.restore(sess,'./weights_dqn_new_task/model_640000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task/model_750000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task_continue/model_1790000.ckpt')
    # saver.restore(sess,'./weights_comb_new_task_new_action(copy)/model_1070000.ckpt')
    for check in range(80):
        saver.restore(sess,'./weights_20181015_2/model_%d.ckpt'%(1000*(check+1)))
        # saver.restore(sess,'./weights_20181015_2/model_23000.ckpt')
        count = 0
        # with open('rewards_comb_new_action.txt','w') as fp:
        for i in range(1000):
            finished = False
            state_list = []
            env.randominit_crowded()
            step = 0

            total_reward = 0
            while not finished and step < 20:
                    step += 1
                    cur_state = env.getstate_1()
                    state_list.append(cur_state)

                    if len(state_list) < frame_num:
                        dif = frame_num - len(state_list)
                        tocon = []
                        for j in range(dif):
                            tocon.append(np.zeros_like(cur_state))
                        
                        state = np.concatenate([*tocon,*state_list], -1)
                        
                    else:
                        state = np.concatenate(state_list[-frame_num:],-1)

                    Qs = sess.run(net.output, feed_dict = {net.inputs_: state.reshape((1, *state.shape))})
                    Qs = Qs.squeeze()
                    
                    cnt = 0
                    while cnt < action_space*obj_num:
                        action = np.argmax(Qs)
                        cnt += 1
                        choice_index = int(action/obj_num)
                        choice_action = action%obj_num

                        reward, done = env.move(choice_index, choice_action)
                        total_reward += reward
                        # state = env.getstate()
                        # mm = env.getmap()
                        # img = convert(mm)
                        # img = Image.fromarray(np.uint8(img))
                        # img.save(os.path.join(save_path,'exp_%d_step_%d.png'%(i,step)))
                        if done == -1:
                            Qs[action] = -100000000
                            continue
                        
                        finished = done
                        break
                    
            if finished:
                count += 1

        print((check+1)*1000,'success rate',count/1000)
                # fp.write('%d %d\n'%(i,total_reward))

def test_new_action_5_frames_add_action():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    import imageio
    save_path = 'test_add_action_frames'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 5
    obj_num = 5
    action_space = 5

    state_size = [15,15,(2*obj_num+1)*frame_num]


    '''
        Setup DQN
    '''
    net = DQNetwork4(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space, frame_num=frame_num)
    
    saver = tf.train.Saver()

    env_test = ENV_scene_new_action_pre_state(size=(15,15))
    sess = tf.Session(config=config)
    # saver.restore(sess,'./weights_20181022_1/model_7720000.ckpt')
    saver.restore(sess,'./weights_20181022_3/model_5940000.ckpt')
    for i in range(20):
        env_test.randominit_crowded()
        state_list_test = []
        action_list_test = []
        finished_test = False
        total_reward = 0
        s = 0
        frames = []
        input_map = env_test.getmap()
        target_map = env_test.gettargetmap()

        img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
        frames.append(img)

        while not finished_test and s < 30:
            s += 1
            state = env_test.getstate_1()

            input_map = np.array(env_test.getmap())
            img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
            frames.append(img)
            
            state_list_test.append(state)

            if len(state_list_test) < frame_num:
                dif = frame_num - len(state_list_test)
                tocon = []
                for j in range(dif):
                    tocon.append(np.zeros_like(state))

                cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                    
            else:
                cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


            if len(action_list_test) < frame_num-1:
                dif = frame_num - len(action_list_test) - 1
                tocon = []
                for j in range(dif):
                    tocon.append(np.zeros(action_space * obj_num))
                
                cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
            else:
                cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

            Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

            Qs = Qs.squeeze()

            while True:
                action = np.argmax(Qs)
                choice_index = int(action / action_space)
                choice_action = action % action_space
                reward, done = env_test.move(choice_index, choice_action)
                total_reward += reward
                if done == -1:
                    Qs[action] = -1000000000
                    continue
                
                finished_test = done
                break

            action_index = action
            action = np.zeros(action_space * obj_num)
            action[action_index] = 1
            action_list_test.append(action)

            choice_action = action_index % action_space
            if choice_action == 4:
                route = env_test.getlastroute()
                route_map = np.zeros_like(input_map)
                for node in route:
                    x,y = node
                    route_map[x,y] = 1
                
                img = convert_to_img(input_map,target_map,route_map)
                frames.append(img)
            
            input_map = env_test.getmap()
            img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
            frames.append(img)
        
        # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
        for j,img in enumerate(frames):
            im = Image.fromarray(img)
            im.save('%s/case%d_%d.png'%(save_path,i,j))

def test_new_action_5_frames_add_action_all_rewards():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    import imageio
    # save_path = 'test_1029_2_193'
    # save_path = 'test_1105_4_245'
    # save_path = 'test_1105_4_276'
    # save_path = 'test_1102_1_356'
    # save_path = 'test_1106_1_245'
    # save_path = 'test_1106_2_245'

    # if not os.path.exists(save_path):
        # os.makedirs(save_path)

    '''
        HyperParameters
    '''

    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)
    
    case_num = len(test_configs)
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 5
    obj_num = 5
    action_space = 5

    state_size = [15,15,(2*obj_num+1)*frame_num]

    weight_paths = []
    weight_paths.append('./weights_20181029_2/model_193000.ckpt')
    # weight_paths.append('./weights_20181105_4/model_276000.ckpt')
    weight_paths.append('./weights_20181102_1/model_356000.ckpt')
    weight_paths.append('./weights_20181105_4/model_245000.ckpt')
    weight_paths.append('./weights_20181106_1/model_245000.ckpt')
    weight_paths.append('./weights_20181106_2/model_245000.ckpt')

    save_matrix = np.zeros([case_num+1, len(weight_paths)])
    

    '''
        Setup DQN
    '''
    net = DQNetwork5(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space, frame_num=frame_num)
    
    saver = tf.train.Saver()

    env_test = ENV_scene_new_action_pre_state_penalty(size=(15,15))
    sess = tf.Session(config=config)
    # saver.restore(sess,'./weights_20181022_1/model_7720000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_273000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_193000.ckpt')
    # saver.restore(sess,'./weights_20181105_4/model_245000.ckpt')
    # saver.restore(sess,'./weights_20181105_4/model_276000.ckpt')
    # saver.restore(sess,'./weights_20181102_1/model_356000.ckpt')
    # saver.restore(sess,'./weights_20181106_1/model_245000.ckpt')
    # saver.restore(sess,'./weights_20181106_2/model_245000.ckpt')

    for idx, path in enumerate(weight_paths):
        saver.restore(sess, path)
        cnt = 0
        print('load',path)
        for i, test_config in enumerate(test_configs):
            print('case',i)
            pos_,target_,size_ = deepcopy(test_config)
            env_test.setmap(pos_,target_,size_)
            # env_test.randominit_crowded()
            state_list_test = []
            action_list_test = []
            finished_test = False
            total_reward = 0
            s = 0
            frames = []
            input_map = env_test.getmap()
            target_map = env_test.gettargetmap()

            # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
            # frames.append(img)

            while not finished_test and s < 100:
                s += 1
                state = env_test.getstate_1()

                input_map = np.array(env_test.getmap())
                # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)
                
                state_list_test.append(state)

                if len(state_list_test) < frame_num:
                    dif = frame_num - len(state_list_test)
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros_like(state))

                    cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                        
                else:
                    cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                if len(action_list_test) < frame_num-1:
                    dif = frame_num - len(action_list_test) - 1
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros(action_space * obj_num))
                    
                    cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                else:
                    cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                Qs = Qs.squeeze()

                while True:
                    action = np.argmax(Qs)
                    choice_index = int(action / action_space)
                    choice_action = action % action_space
                    reward, done = env_test.move(choice_index, choice_action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

                action_index = action
                action = np.zeros(action_space * obj_num)
                action[action_index] = 1
                action_list_test.append(action)

                choice_action = action_index % action_space
                if choice_action == 4:
                    route = env_test.getlastroute()
                    route_map = np.zeros_like(input_map)
                    for node in route:
                        x,y = node
                        route_map[x,y] = 1
                    
                    # img = convert_to_img(input_map,target_map,route_map)
                    # frames.append(img)
                
                input_map = env_test.getmap()
                # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)
            
            if finished_test:
                cnt += 1
                save_matrix[i, idx] = 1
                print('finish!')
            
            # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
            # for j,img in enumerate(frames):
            #     im = Image.fromarray(img)
            #     im.save('%s/case%d_%d.png'%(save_path,i,j))

        save_matrix[case_num, idx] = cnt
        cnt = 1.0*cnt / len(test_configs)
        
        # with open('%s/sr.txt'%save_path,'w') as fp:
        #     fp.write('%f\n'%cnt)
    
    with open('evaluation_100steps.txt','w') as fp:
        for i, case in enumerate(save_matrix):
            fp.write('case %d: '%(i+1))
            tot = 0
            for _ in case:
                fp.write('%d '%_)
                tot += _
            
            fp.write('sum %d\n'%tot)

def test_new_action_5_frames_add_action_all_rewards_conflict():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    import imageio
    # save_path = 'test_1029_2_193'
    # save_path = 'test_1105_4_245'
    # save_path = 'test_1105_4_276'
    # save_path = 'test_1102_1_356'
    # save_path = 'test_1106_1_245'
    # save_path = 'test_1106_2_245'
    save_paths = []
    save_paths.append('test_1110_3_252')
    save_paths.append('test_1110_3_326')
    for save_path in save_paths:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    '''
        HyperParameters
    '''

    with open('test.pkl','rb') as fp:
        test_configs = pickle.load(fp)
    
    case_num = len(test_configs)
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 5
    obj_num = 5
    action_space = 5

    state_size = [15,15,(2*obj_num+1)*frame_num]

    weight_paths = []
    # weight_paths.append('./weights_20181029_2/model_193000.ckpt')
    # weight_paths.append('./weights_20181105_4/model_276000.ckpt')
    # weight_paths.append('./weights_20181102_1/model_356000.ckpt')
    # weight_paths.append('./weights_20181105_4/model_245000.ckpt')
    # weight_paths.append('./weights_20181106_1/model_245000.ckpt')
    # weight_paths.append('./weights_20181106_2/model_245000.ckpt')

    weight_paths.append('./weights_20181110_3/model_252000.ckpt')
    # weight_paths.append('./weights_20181110_3/model_326000.ckpt')

    save_matrix = np.zeros([case_num+1, len(weight_paths)])
    

    '''
        Setup DQN
    '''
    net = DQNetwork7(state_size=state_size,learning_rate=learning_rate,action_space=action_space, num_objects=obj_num, frame_num=frame_num)
    
    saver = tf.train.Saver()

    env_test = ENV_scene_new_action_pre_state_penalty_conflict(size=(15,15))
    sess = tf.Session(config=config)
    # saver.restore(sess,'./weights_20181022_1/model_7720000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_273000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_193000.ckpt')
    # saver.restore(sess,'./weights_20181105_4/model_245000.ckpt')
    # saver.restore(sess,'./weights_20181105_4/model_276000.ckpt')
    # saver.restore(sess,'./weights_20181102_1/model_356000.ckpt')
    # saver.restore(sess,'./weights_20181106_1/model_245000.ckpt')
    # saver.restore(sess,'./weights_20181106_2/model_245000.ckpt')
    total_copy_time = 0
    total_forward_time = 0
    total_nn_time = 0
    total_step = 0
    total_move_time = 0
    total_move = 0
    for idx, path in enumerate(weight_paths):
        saver.restore(sess, path)
        cnt = 0
        print('load',path)
        for i, test_config in enumerate(test_configs):
            print('case',i)
            t = time.time()
            pos_,target_,size_ = deepcopy(test_config)
            env_test.setmap(pos_,target_,size_)
            total_copy_time += time.time()-t
            # env_test.randominit_crowded()
            state_list_test = []
            action_list_test = []
            finished_test = False
            total_reward = 0
            s = 0
            # frames = []
            input_map = env_test.getmap()
            target_map = env_test.gettargetmap()

            # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
            # frames.append(img)

            while not finished_test and s < 100:
                s += 1
                t = time.time()
                state = env_test.getstate_1()

                input_map = np.array(env_test.getmap())
                # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)
                conflict_matrix = env_test.getconflict()
                finish_tag = env_test.getfinished()
                state_list_test.append(state)

                if len(state_list_test) < frame_num:
                    dif = frame_num - len(state_list_test)
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros_like(state))

                    cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                        
                else:
                    cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                if len(action_list_test) < frame_num-1:
                    dif = frame_num - len(action_list_test) - 1
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros(action_space * obj_num))
                    
                    cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                else:
                    cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)
                t_ = time.time()
                Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape)), net.conflict_matrix: conflict_matrix.reshape((1, *conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1,*finish_tag.shape))})
                total_nn_time += time.time() - t_
                Qs = Qs.squeeze()

                while True:
                    action = np.argmax(Qs)
                    choice_index = int(action / action_space)
                    choice_action = action % action_space
                    t__ = time.time()
                    reward, done = env_test.move(choice_index, choice_action)
                    total_move_time += time.time() - t__
                    total_move += 1
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break
                
                action_index = action
                action = np.zeros(action_space * obj_num)
                action[action_index] = 1
                action_list_test.append(action)

                choice_action = action_index % action_space
                if choice_action == 4:
                    route = env_test.getlastroute()
                    route_map = np.zeros_like(input_map)
                    for node in route:
                        x,y = node
                        route_map[x,y] = 1
                    
                    # img = convert_to_img(input_map,target_map,route_map)
                    # frames.append(img)
                
                input_map = env_test.getmap()
                # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)
                total_forward_time += time.time()-t


            total_step += s

            if finished_test:
                cnt += 1
                save_matrix[i, idx] = 1
                print('finish!')
            
            # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
            # for j,img in enumerate(frames):
            #     im = Image.fromarray(img)
            #     im.save('%s/case%d_%d.png'%(save_paths[idx],i,j))

        save_matrix[case_num, idx] = cnt
        cnt = 1.0*cnt / len(test_configs)
        
        # with open('%s/sr.txt'%save_path,'w') as fp:
        #     fp.write('%f\n'%cnt)
    print('mean copy time',total_copy_time/200,'mean forward time',total_forward_time/total_step,'mean nn time',total_nn_time/total_step,'mean move time',total_move_time/total_move)
    # with open('evaluation_100steps_conflict.txt','w') as fp:
    #     for i, case in enumerate(save_matrix):
    #         fp.write('case %d: '%(i+1))
    #         tot = 0
    #         for _ in case:
    #             fp.write('%d '%_)
    #             tot += _
            
    #         fp.write('sum %d\n'%tot)
    #     _list = [case[len(weight_paths)-1] for case in save_matrix]
    #     tot = 0
    #     for _ in _list:
    #         tot += (_ > 0)
    #     fp.write('total %d\n'%tot)

def test_new_action_one_frame_add_action_all_rewards_MCTS():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    from MCTS import MCT
    import imageio
    # from math import pow
    # save_path = 'test_1029_2_193'
    # save_path = 'test_1105_4_245'
    # save_path = 'test_1105_4_276'
    # save_path = 'test_1102_1_356'
    # save_path = 'test_1106_1_245'
    # save_path = 'NN_MCTS_100_sim'
    # save_path = 'NN_MCTS_100_sim_discount'
    # save_path = 'NN_MCTS_50_sim_100_random_discount'
    # save_path = 'NN_MCTS_100_random_50_sim_discount'
    save_path = 'NN_MCTS_50NN_50random_sim_discount'
    gamma = 0.95
    
    exps = [17]
    # illist = [30, 46, 73, 109, 116, 130, 132, 138, 186]
    illist = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        HyperParameters
    '''
    
        
        
    # case_num = len(test_configs)
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 1
    obj_num = 17
    action_space = 5
    action_size = 17 * action_space

    state_size = [35,35,(2*17+1)*frame_num]

    weight_paths = []

    # weight_paths.append('./weights_20181128_1/model_20000.ckpt')
    weight_paths.append('./weights_20190103_1/model_36600.ckpt')

    # save_matrix = np.zeros([case_num+1, len(weight_paths)])
    

    '''
        Setup DQN
    '''
    net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    saver = tf.train.Saver()

    # env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(15,15))
    env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(35,35),max_num=obj_num)
    sess = tf.Session(config=config)

    for idx, path in enumerate(weight_paths):
        saver.restore(sess, path)
        cnt = 0
        print('load',path)

        for exp in exps:
            steps = []
            times = []
            rewards_cases = []
            with open('exp35_%d.pkl'%exp,'rb') as fp:
                test_configs = pickle.load(fp)
            

            for idd, test_config in enumerate(test_configs):
                if idd in illist:
                    continue
                # if exp == 17 and idd < 7:
                #     continue
                if idd != 19:
                    continue
                if idd == 20:
                    break
                # if idd != 8:
                #     continue

                print('case',idd)
                frames = []
                last_list = []
                pos_,target_,size_ = deepcopy(test_config)
                env_test.setmap(pos_,target_,size_)
                # env_test.randominit_crowded()
                finished_test = False
                total_reward = 0
                s = 0
                input_map = deepcopy(env_test.getmap())
                target_map = env_test.gettargetmap()

                # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)

                tree = MCT()
                tree.root.state = deepcopy(env_test)
                t = time.time()
                tree.setactionsize(action_size)
                cflag = False

                while (not (finished_test==1)) and s < 100:
                    s += 1
                    max_depth = 0
                    
                    for i in range(action_size):
                        if not tree.haschild(tree.root.id, i):
                            item = int(i/action_space)
                            direction = i % action_space
                            env_copy = deepcopy(env_test)
                            reward, done = env_copy.move(item, direction)
                            succ, child_id = tree.expansion(tree.root.id, i, reward, env_copy, done)

                            if succ:
                                if done != 1:
                                    policy = 0
                                    cnt = 0
                                    reward_sum_a = 0
                                    env_sim = deepcopy(env_copy)
                                    end_flag = False
                                    while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        # state = env_sim.getstate_1()
                                        # conflict = env_sim.getconflict()
                                        # finishtag = env_sim.getfinished()
                                        state = env_sim.getstate_1()
                                        state = np.pad(state,((0,0),(0,0),(0,(obj_num-exp)*2)),"constant")

                                        conflict_matrix = env_sim.getconflict()
                                        conflict = np.pad(conflict_matrix,((0,obj_num-exp),(0,obj_num-exp),(0,0)),"constant")
                                        finish_tag = env_sim.getfinished()
                                        finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")

                                        Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict.reshape((1, *conflict.shape)), net.finish_tag: finishtag.reshape((1, *finishtag.shape))})
                                        
                                        Qs = Qs.squeeze()

                                        while True:
                                            action = np.argmax(Qs)
                                            item = int(action / action_space)
                                            direction = action % action_space
                                            reward, done = env_sim.move(item, direction)

                                            if done != -1:
                                                break
                                            
                                            Qs[action] = -10000000
                                        
                                        reward_sum_a += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                    
                                    reward_sum_b = 0
                                    cnt = 0
                                    last_item = -1
                                    end_flag = False
                                    env_sim = deepcopy(env_copy)
                                    while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue
                                                direction = np.random.randint(4)
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    break

                                        reward_sum_b += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                
                                    
                                    reward_sum_c = 0
                                    cnt = 0
                                    last_item = -1
                                    end_flag = False
                                    env_sim = deepcopy(env_copy)
                                    while (not (end_flag == 1)) and cnt < 20:
                                            cnt += 1
                                            jumps = []
                                            for i in range(exp):
                                                jumps.append((-size_[i][0]*size_[i][1],i))
                                        
                                            sorted(jumps)
                                            
                                            cmax = -1
                                            item = -1
                                            direction = 4

                                            for i in jumps:
                                                reward, done = env_sim.move(i[1],4)
                                                if done != -1:
                                                    item = i[1]
                                                    break
            

                                            if item == -1:
                                                while True:
                                                    item = np.random.randint(exp)
                                                    if item == last_item:
                                                        continue

                                                    direction = np.random.randint(2)
                                                    direction = direction * 2
                                                    
                                                    reward, done = env_sim.move(item, direction)
                                                    if done != -1:
                                                        break

                                            reward_sum_c += reward * pow(gamma, cnt-1)
                                            end_flag = done
                                    
                                    
                                    if reward_sum_a < reward_sum_b and reward_sum_b > reward_sum_c:
                                        reward_sum = reward_sum_b
                                        policy = 1
                                    elif reward_sum_a > reward_sum_b and reward_sum_a > reward_sum_c:
                                        reward_sum = reward_sum_a
                                        policy = 0
                                    else:
                                        reward_sum = reward_sum_c
                                        policy = 2

                                    tree.nodedict[child_id].value = reward_sum
                                    tree.nodedict[child_id].policy = policy

                                    if reward_sum > 6000:
                                        print('wa done!',reward_sum,'policy',policy)
                                        cflag = True

                                tree.backpropagation(child_id)

                    if cflag:
                        C = 1.0
                    else:
                        C = 3.0
                    
                    cflag = False
                    print(C)
                    
                    t_nn = time.time()
                    total_io = 0
                    for __ in range(200):
                        _id = tree.selection(C)
                        policy = tree.nodedict[_id].policy
                        env_copy = deepcopy(tree.getstate(_id))
                        if policy == 0:
                        
                        
                            # state = env_copy.getstate_1()
                            # conflict = env_copy.getconflict()
                            # finishtag = env_copy.getfinished()
                            
                            state = env_copy.getstate_1()
                            io_ = time.time()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-exp)*2)),"constant")
                            total_io += time.time() - io_

                            
                            conflict_matrix = env_copy.getconflict()
                            conflict = np.pad(conflict_matrix,((0,obj_num-exp),(0,obj_num-exp),(0,0)),"constant")
                            finish_tag = env_copy.getfinished()
                            finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict.reshape((1, *conflict.shape)), net.finish_tag: finishtag.reshape((1, *finishtag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                if tree.haschild(_id, action):
                                    Qs[action] = -1000000
                                    continue
                                
                                break
                        
                        elif policy == 1:
                            jumps = []
                            for i in range(exp):
                                jumps.append((-size_[i][0]*size_[i][1],i))
                            
                            sorted(jumps)

                            env_copy_= deepcopy(env_copy)
                            action = -1
                            for i in jumps:
                                reward, done = env_copy_.move(i[1],4)
                                if done != -1:
                                    # item = i[1]
                                    action = i[1]*action_space+4
                                    break

                            if action == -1:
                                action = np.random.randint(action_size)
                        else:
                            jumps = []
                            for i in range(exp):
                                jumps.append((-size_[i][0]*size_[i][1],i))
                            
                            sorted(jumps)

                            env_copy_= deepcopy(env_copy)
                            action = -1
                            for i in jumps:
                                reward, done = env_copy_.move(i[1],4)
                                if done != -1:
                                    # item = i[1]
                                    action = i[1]*action_space+4
                                    break

                            if action == -1:
                                item = np.random.randint(exp)
                                direction = np.random.randint(2)*2
                                action = item*action_space+direction

                        while True:
                            actions = tree.getemptyactions(_id)
                            if not action in actions:
                                action = actions[np.random.randint(len(actions))]
                                continue

                            item = int(action / action_space)
                            direction = action % action_space
                            reward, done = env_copy.move(item, direction)
                            
                            succ, child_id = tree.expansion(_id, action, reward, env_copy, done)
                            break

                        if succ:
                            if done != 1:
                                state = env_copy.getstate_1()
                                conflict = env_copy.getconflict()
                                finishtag = env_copy.getfinished()

                                # value = np.max(Qs)
                                cnt = 0
                                reward_sum_a = 0
                                env_sim = deepcopy(env_copy)
                                end_flag = False
                                while (not( end_flag == 1 )) and cnt < 20:
                                    cnt += 1
                                    # state = env_sim.getstate_1()
                                    # conflict = env_sim.getconflict()
                                    # finishtag = env_sim.getfinished()
                                    io_ = time.time()
                                    state = env_sim.getstate_1()
                                    state = np.pad(state,((0,0),(0,0),(0,(obj_num-exp)*2)),"constant")

                                    total_io += time.time() - io_


                                    conflict_matrix = env_sim.getconflict()
                                    conflict = np.pad(conflict_matrix,((0,obj_num-exp),(0,obj_num-exp),(0,0)),"constant")
                                    finish_tag = env_sim.getfinished()
                                    finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")

                                    Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict.reshape((1, *conflict.shape)), net.finish_tag: finishtag.reshape((1, *finishtag.shape))})

                                    Qs = Qs.squeeze()

                                    while True:
                                        action = np.argmax(Qs)
                                        item = int(action / action_space)
                                        direction = action % action_space
                                        reward, done = env_sim.move(item, direction)

                                        if done != -1:
                                            break
                                        
                                        Qs[action] = -10000000
                                    
                                    reward_sum_a += reward * pow(gamma, cnt-1)
                                    end_flag = done
                                
                                # if end_flag == 1:
                                #     print('wa done!',reward_sum)

                                reward_sum_b = 0
                                cnt = 0
                                last_item = -1
                                end_flag = False
                                env_sim = deepcopy(env_copy)
                                while (not( end_flag == 1 )) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue
                                                direction = np.random.randint(4)
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    break

                                        reward_sum_b += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                # tree.nodedict[child_id].value = value
                                
                                reward_sum_c = 0
                                cnt = 0
                                last_item = -1
                                end_flag = False
                                env_sim = deepcopy(env_copy)
                                while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue

                                                direction = np.random.randint(2)
                                                direction = direction * 2
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    break

                                        reward_sum_c += reward * pow(gamma, cnt-1)
                                        end_flag = done

                                if reward_sum_a < reward_sum_b and reward_sum_b > reward_sum_c:
                                    reward_sum = reward_sum_b
                                    policy = 1
                                elif reward_sum_a > reward_sum_b and reward_sum_a > reward_sum_c:
                                    reward_sum = reward_sum_a
                                    policy = 0
                                else:
                                    reward_sum = reward_sum_c
                                    policy = 2

                                tree.nodedict[child_id].value = reward_sum
                                tree.nodedict[child_id].policy = policy
                                if reward_sum > 6000:
                                    print('wa done!',reward_sum,'policy',policy)
                                    cflag = True

                            max_depth = max([max_depth, tree.nodedict[child_id].depth])
                            tree.backpropagation(child_id)

                    t_nn = time.time() - t_nn                    
                    action = tree.root.best
                    item = int(action / action_space)
                    direction = action % action_space
                    reward, done = env_test.move(item, direction)
                    finished_test = done
                    
                    print('step',s,'times',tree.nodedict[tree.root.childs[action]].times,'id',tree.root.id, 'max_depth',max_depth-tree.root.depth, 'value',tree.root.value,'best depth',tree.getbestdepth(),'io time',total_io,'nn time',t_nn,'policy',tree.root.policy,'current reward',total_reward)
                    # print('last')
                    # for tid in last_list:
                    #     node = tree.nodedict[tid]
                    #     print('id',node.id,'value',node.value,'reward',node.reward,'best',node.best,'depth',node.depth-tree.root.depth)

                    last_list = []
                    node = tree.root
                    best = node.best
                    print('this')
                    while best != -1:
                        node = tree.nodedict[node.childs[best]]
                        best = node.best
                        print('id',node.id,'value',node.value,'reward',node.reward,'best',best,'depth',node.depth-tree.root.depth)
                        last_list.append(node.id)
                        

                    tree.nextstep(action)
                    total_reward += reward
                    
                    # if direction == 4:
                    #     route = env_test.getlastroute()
                    #     route_map = np.zeros_like(input_map)
                    #     for node in route:
                    #         x,y = node
                    #         route_map[x,y] = 1
                        
                    #     img = convert_to_img(input_map,target_map,route_map)
                    #     frames.append(img)
                    
                    
                    # input_map = deepcopy(env_test.getmap())
                    # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                    # frames.append(img)
                
                times.append(time.time()-t)
                steps.append(s)
                rewards_cases.append(total_reward)

                if finished_test == 1:
                    print(idd,'Finished! Reward:',total_reward,'Steps:',s)
                else:
                    print(idd,'Failed Reward:',total_reward,'Steps:',s)
                
                # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
                # for j,img in enumerate(frames):
                #     im = Image.fromarray(img)
                #     im.save('%s/case%d_%d.png'%(save_path,idd,j))

            # with open('exp_%d_step.txt'%exp,'w') as fp:
            #     for st in steps:
            #         fp.write('%d\n'%st)
            # with open('exp_%d_time.txt'%exp,'w') as fp:
            #     for t in times:
            #         fp.write('%f\n'%t)
            # with open('exp_%d_reward.txt'%exp,'w') as fp:
            #     for r in rewards_cases:
            #         fp.write('%d\n'%r)

            # print('exp',exp,'mean step',np.array(steps).mean(),'mean time',np.array(times).mean())

def test_new_action_one_frame_add_action_all_rewards_MCTS_transpose_shape():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    from MCTS import MCT
    import imageio
    # from math import pow
    # save_path = 'test_1029_2_193'
    # save_path = 'test_1105_4_245'
    # save_path = 'test_1105_4_276'
    # save_path = 'test_1102_1_356'
    # save_path = 'test_1106_1_245'
    # save_path = 'NN_MCTS_100_sim'
    # save_path = 'NN_MCTS_100_sim_discount'
    # save_path = 'NN_MCTS_50_sim_100_random_discount'
    # save_path = 'NN_MCTS_100_random_50_sim_discount'
    save_path = 'NN_MCTS_50NN_50random_sim_discount'
    gamma = 0.95
    
    exps = [17]
    # illist = [30, 46, 73, 109, 116, 130, 132, 138, 186]
    illist = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        HyperParameters
    '''
    
        
        
    # case_num = len(test_configs)
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 1
    obj_num = 17
    action_space = 5
    action_size = 17 * action_space

    state_size = [35,35,(2*17+1)*frame_num]

    weight_paths = []

    # weight_paths.append('./weights_20181128_1/model_20000.ckpt')
    weight_paths.append('./weights_20190103_1/model_36600.ckpt')

    # save_matrix = np.zeros([case_num+1, len(weight_paths)])
    

    '''
        Setup DQN
    '''
    net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    saver = tf.train.Saver()

    # env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(15,15))
    env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(35,35),max_num=obj_num)
    sess = tf.Session(config=config)

    for idx, path in enumerate(weight_paths):
        saver.restore(sess, path)
        cnt = 0
        print('load',path)

        for exp in exps:
            steps = []
            times = []
            rewards_cases = []
            with open('exp35_%d.pkl'%exp,'rb') as fp:
                test_configs = pickle.load(fp)
            

            for idd, test_config in enumerate(test_configs):
                if idd in illist:
                    continue
                # if exp == 17 and idd < 7:
                #     continue
                if idd != 19:
                    continue
                if idd == 20:
                    break
                # if idd != 8:
                #     continue

                print('case',idd)
                frames = []
                last_list = []
                pos_,target_,size_ = deepcopy(test_config)
                env_test.setmap(pos_,target_,size_)
                # env_test.randominit_crowded()
                finished_test = False
                total_reward = 0
                s = 0
                input_map = deepcopy(env_test.getmap())
                target_map = env_test.gettargetmap()

                img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                frames.append(img)
                

                tree = MCT()
                tree.root.state = deepcopy(env_test)
                t = time.time()
                tree.setactionsize(action_size)
                cflag = False

                while (not (finished_test==1)) and s < 100:
                    s += 1
                    max_depth = 0
                    
                    for i in range(action_size):
                        if not tree.haschild(tree.root.id, i):
                            item = int(i/action_space)
                            direction = i % action_space
                            env_copy = deepcopy(env_test)
                            reward, done = env_copy.move(item, direction)
                            succ, child_id = tree.expansion(tree.root.id, i, reward, env_copy, done)

                            if succ:
                                if done != 1:
                                    policy = 0
                                    cnt = 0
                                    reward_sum_a = 0
                                    env_sim = deepcopy(env_copy)
                                    end_flag = False
                                    while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        # state = env_sim.getstate_1()
                                        # conflict = env_sim.getconflict()
                                        # finishtag = env_sim.getfinished()
                                        state = env_sim.getstate_1()
                                        state = np.pad(state,((0,0),(0,0),(0,(obj_num-exp)*2)),"constant")

                                        conflict_matrix = env_sim.getconflict()
                                        conflict = np.pad(conflict_matrix,((0,obj_num-exp),(0,obj_num-exp),(0,0)),"constant")
                                        finish_tag = env_sim.getfinished()
                                        finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")

                                        Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict.reshape((1, *conflict.shape)), net.finish_tag: finishtag.reshape((1, *finishtag.shape))})
                                        
                                        Qs = Qs.squeeze()

                                        while True:
                                            action = np.argmax(Qs)
                                            item = int(action / action_space)
                                            direction = action % action_space
                                            reward, done = env_sim.move(item, direction)

                                            if done != -1:
                                                break
                                            
                                            Qs[action] = -10000000
                                        
                                        reward_sum_a += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                    
                                    reward_sum_b = 0
                                    cnt = 0
                                    last_item = -1
                                    end_flag = False
                                    env_sim = deepcopy(env_copy)
                                    while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue
                                                direction = np.random.randint(4)
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    break

                                        reward_sum_b += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                
                                    
                                    reward_sum_c = 0
                                    cnt = 0
                                    last_item = -1
                                    end_flag = False
                                    env_sim = deepcopy(env_copy)
                                    while (not (end_flag == 1)) and cnt < 20:
                                            cnt += 1
                                            jumps = []
                                            for i in range(exp):
                                                jumps.append((-size_[i][0]*size_[i][1],i))
                                        
                                            sorted(jumps)
                                            
                                            cmax = -1
                                            item = -1
                                            direction = 4

                                            for i in jumps:
                                                reward, done = env_sim.move(i[1],4)
                                                if done != -1:
                                                    item = i[1]
                                                    break
            

                                            if item == -1:
                                                while True:
                                                    item = np.random.randint(exp)
                                                    if item == last_item:
                                                        continue

                                                    direction = np.random.randint(2)
                                                    direction = direction * 2
                                                    
                                                    reward, done = env_sim.move(item, direction)
                                                    if done != -1:
                                                        break

                                            reward_sum_c += reward * pow(gamma, cnt-1)
                                            end_flag = done
                                    
                                    
                                    if reward_sum_a < reward_sum_b and reward_sum_b > reward_sum_c:
                                        reward_sum = reward_sum_b
                                        policy = 1
                                    elif reward_sum_a > reward_sum_b and reward_sum_a > reward_sum_c:
                                        reward_sum = reward_sum_a
                                        policy = 0
                                    else:
                                        reward_sum = reward_sum_c
                                        policy = 2

                                    tree.nodedict[child_id].value = reward_sum
                                    tree.nodedict[child_id].policy = policy

                                    if reward_sum > 6000:
                                        print('wa done!',reward_sum,'policy',policy)
                                        cflag = True

                                tree.backpropagation(child_id)

                    if cflag:
                        C = 1.0
                    else:
                        C = 3.0
                    
                    cflag = False
                    print('C=',C)
                    
                    t_nn = time.time()
                    total_io = 0
                    for __ in range(200):
                        _id = tree.selection(C)
                        policy = tree.nodedict[_id].policy
                        env_copy = deepcopy(tree.getstate(_id))
                        if policy == 0:
                        
                        
                            # state = env_copy.getstate_1()
                            # conflict = env_copy.getconflict()
                            # finishtag = env_copy.getfinished()
                            
                            state = env_copy.getstate_1()
                            io_ = time.time()
                            state = np.pad(state,((0,0),(0,0),(0,(obj_num-exp)*2)),"constant")
                            total_io += time.time() - io_

                            
                            conflict_matrix = env_copy.getconflict()
                            conflict = np.pad(conflict_matrix,((0,obj_num-exp),(0,obj_num-exp),(0,0)),"constant")
                            finish_tag = env_copy.getfinished()
                            finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")

                            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict.reshape((1, *conflict.shape)), net.finish_tag: finishtag.reshape((1, *finishtag.shape))})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                if tree.haschild(_id, action):
                                    Qs[action] = -1000000
                                    continue
                                
                                break
                        
                        elif policy == 1:
                            jumps = []
                            for i in range(exp):
                                jumps.append((-size_[i][0]*size_[i][1],i))
                            
                            sorted(jumps)

                            env_copy_= deepcopy(env_copy)
                            action = -1
                            for i in jumps:
                                reward, done = env_copy_.move(i[1],4)
                                if done != -1:
                                    # item = i[1]
                                    action = i[1]*action_space+4
                                    break

                            if action == -1:
                                action = np.random.randint(action_size)
                        else:
                            jumps = []
                            for i in range(exp):
                                jumps.append((-size_[i][0]*size_[i][1],i))
                            
                            sorted(jumps)

                            env_copy_= deepcopy(env_copy)
                            action = -1
                            for i in jumps:
                                reward, done = env_copy_.move(i[1],4)
                                if done != -1:
                                    # item = i[1]
                                    action = i[1]*action_space+4
                                    break

                            if action == -1:
                                item = np.random.randint(exp)
                                direction = np.random.randint(2)*2
                                action = item*action_space+direction

                        while True:
                            actions = tree.getemptyactions(_id)
                            if not action in actions:
                                action = actions[np.random.randint(len(actions))]
                                continue

                            item = int(action / action_space)
                            direction = action % action_space
                            reward, done = env_copy.move(item, direction)
                            
                            succ, child_id = tree.expansion(_id, action, reward, env_copy, done)
                            break

                        if succ:
                            if done != 1:
                                state = env_copy.getstate_1()
                                conflict = env_copy.getconflict()
                                finishtag = env_copy.getfinished()

                                # value = np.max(Qs)
                                cnt = 0
                                reward_sum_a = 0
                                env_sim = deepcopy(env_copy)
                                end_flag = False
                                while (not( end_flag == 1 )) and cnt < 20:
                                    cnt += 1
                                    # state = env_sim.getstate_1()
                                    # conflict = env_sim.getconflict()
                                    # finishtag = env_sim.getfinished()
                                    io_ = time.time()
                                    state = env_sim.getstate_1()
                                    state = np.pad(state,((0,0),(0,0),(0,(obj_num-exp)*2)),"constant")

                                    total_io += time.time() - io_


                                    conflict_matrix = env_sim.getconflict()
                                    conflict = np.pad(conflict_matrix,((0,obj_num-exp),(0,obj_num-exp),(0,0)),"constant")
                                    finish_tag = env_sim.getfinished()
                                    finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")

                                    Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict.reshape((1, *conflict.shape)), net.finish_tag: finishtag.reshape((1, *finishtag.shape))})

                                    Qs = Qs.squeeze()

                                    while True:
                                        action = np.argmax(Qs)
                                        item = int(action / action_space)
                                        direction = action % action_space
                                        reward, done = env_sim.move(item, direction)

                                        if done != -1:
                                            break
                                        
                                        Qs[action] = -10000000
                                    
                                    reward_sum_a += reward * pow(gamma, cnt-1)
                                    end_flag = done
                                
                                # if end_flag == 1:
                                #     print('wa done!',reward_sum)

                                reward_sum_b = 0
                                cnt = 0
                                last_item = -1
                                end_flag = False
                                env_sim = deepcopy(env_copy)
                                while (not( end_flag == 1 )) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue
                                                direction = np.random.randint(4)
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    break

                                        reward_sum_b += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                # tree.nodedict[child_id].value = value
                                
                                reward_sum_c = 0
                                cnt = 0
                                last_item = -1
                                end_flag = False
                                env_sim = deepcopy(env_copy)
                                while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue

                                                direction = np.random.randint(2)
                                                direction = direction * 2
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    break

                                        reward_sum_c += reward * pow(gamma, cnt-1)
                                        end_flag = done

                                if reward_sum_a < reward_sum_b and reward_sum_b > reward_sum_c:
                                    reward_sum = reward_sum_b
                                    policy = 1
                                elif reward_sum_a > reward_sum_b and reward_sum_a > reward_sum_c:
                                    reward_sum = reward_sum_a
                                    policy = 0
                                else:
                                    reward_sum = reward_sum_c
                                    policy = 2

                                tree.nodedict[child_id].value = reward_sum
                                tree.nodedict[child_id].policy = policy
                                if reward_sum > 6000:
                                    print('wa done!',reward_sum,'policy',policy)
                                    cflag = True

                            max_depth = max([max_depth, tree.nodedict[child_id].depth])
                            tree.backpropagation(child_id)

                    t_nn = time.time() - t_nn                    
                    action = tree.root.best
                    item = int(action / action_space)
                    direction = action % action_space
                    reward, done = env_test.move(item, direction)
                    finished_test = done
                    
                    print('step',s,'times',tree.nodedict[tree.root.childs[action]].times,'id',tree.root.id, 'max_depth',max_depth-tree.root.depth, 'value',tree.root.value,'best depth',tree.getbestdepth(),'io time',total_io,'nn time',t_nn,'policy',tree.root.policy,'current reward',total_reward)

                    if direction == 4:
                        route = env_test.getlastroute()
                        route_map = np.zeros_like(input_map)
                        for node in route:
                            x,y,s,ds = node
                            route_map[x,y] = 1
                        
                        for node in route:
                            x,y,s,ds = node
                            imap = env_test.getcleanmap(item)
                            ps,bx = env_test.getitem(item, s)
                            for p in ps:
                                xx,yy = p
                                xx += x
                                yy += y
                                imap[xx,yy] = item + 2
                            
                            img = convert_to_img(imap, target_map, route_map)
                            frames.append(img)
                        
                    
                    input_map = deepcopy(env_test.getmap())
                    img = convert_to_img(input_map, target_map, np.zeros_like(input_map))
                    frames.append(img)
                    # print('last')
                    # for tid in last_list:
                    #     node = tree.nodedict[tid]
                    #     print('id',node.id,'value',node.value,'reward',node.reward,'best',node.best,'depth',node.depth-tree.root.depth)

                    last_list = []
                    node = tree.root
                    best = node.best
                    print('this')
                    while best != -1:
                        node = tree.nodedict[node.childs[best]]
                        best = node.best
                        print('id',node.id,'value',node.value,'reward',node.reward,'best',best,'depth',node.depth-tree.root.depth)
                        last_list.append(node.id)
                        

                    tree.nextstep(action)
                    total_reward += reward
                    
                    # if direction == 4:
                    #     route = env_test.getlastroute()
                    #     route_map = np.zeros_like(input_map)
                    #     for node in route:
                    #         x,y = node
                    #         route_map[x,y] = 1
                        
                    #     img = convert_to_img(input_map,target_map,route_map)
                    #     frames.append(img)
                    
                    
                    # input_map = deepcopy(env_test.getmap())
                    # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                    # frames.append(img)
                
                times.append(time.time()-t)
                steps.append(s)
                rewards_cases.append(total_reward)

                if finished_test == 1:
                    print(idd,'Finished! Reward:',total_reward,'Steps:',s)
                else:
                    print(idd,'Failed Reward:',total_reward,'Steps:',s)
                
                break


def test_simple_MCTS():
    from MCTS import MCT
    from visulization import convert_to_img
    from PIL import Image

    action_space = 5
    num_obj = 12
    action_size = action_space * num_obj
    save_path = 'simple_MCTS_inherit_100_12obj'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('test35.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(35,35))

    
    for idx, test_config in enumerate(test_configs):
        print('case',idx)
        frames = []
        pos_,target_,size_ = deepcopy(test_config)
        env_test.setmap(pos_,target_,size_)
        finished_test = False
        s = 0
        total_reward = 0
        input_map = deepcopy(env_test.getmap())
        target_map = env_test.gettargetmap()

        img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
        frames.append(img)
        tree = MCT()
        tree.root.state = deepcopy(env_test)

        while not finished_test and s < 100:
            s += 1
            max_depth = 0
            for i in range(action_size):

                if not tree.haschild(tree.root.id, i):
                    env_copy = deepcopy(env_test)
                    item = int(i / action_space)
                    direction = i % action_space
                    reward, done = env_copy.move(item, direction)

                    succ, child_id = tree.expansion(tree.root.id, i, reward, env_copy, done == -1)
                    
                    if succ:
                        tree.nodedict[child_id].value = 0
                        tree.backpropagation(child_id)
            
            for i in range(200):
                _id = tree.selection()
                action = 0
                cnt = 0
                error = True
                while True:
                    action = np.random.randint(action_size)
                    cnt += 1
                    if cnt > 100:
                        error = True
                        for j in range(action_size):
                            if tree.haschild(_id, j):
                                error = False
                        if error:
                            break

                    if not tree.haschild(_id, action):
                        error = False
                        break

                if error:
                    print('expansion remove error')
                    continue
                                    
                item = int(action / action_space)
                direction = action % action_space
                env_copy = deepcopy(tree.getstate(_id))
                reward, done = env_copy.move(item, direction)

                succ, child_id = tree.expansion(_id, action, reward, env_copy, done == -1)

                if succ:
                    tree.nodedict[child_id].value = 0
                    max_depth = max([max_depth, tree.nodedict[child_id].depth])
                    tree.backpropagation(child_id)

            action = tree.root.best
            item = int(action / action_space)
            direction = action % action_space
            reward, done = env_test.move(item, direction)
            finished_test = done
            total_reward += reward
            print('times',tree.nodedict[tree.root.childs[action]].times, 'max_depth',max_depth-tree.root.depth)
            tree.nextstep(action)

            if direction == 4:
                route = env_test.getlastroute()
                route_map = np.zeros_like(input_map)
                for node in route:
                    x,y = node
                    route_map[x,y] = 1
                img = convert_to_img(input_map, target_map, route_map)
                frames.append(img)
            
            input_map = deepcopy(env_test.getmap())
            img = convert_to_img(input_map, target_map, np.zeros_like(input_map))
            frames.append(img)
        
        for j,img in enumerate(frames):
                im = Image.fromarray(img)
                im.save('%s/case%d_%d.png'%(save_path,idx,j))

        if finished_test:
            print('Finished! Reward:',total_reward,'Steps:',s)
        else:
            print('Failed Reward:',total_reward,'Steps:',s)

def test_simple_MCTS_transpose():
    from MCTS import MCT
    from src.visualization import convert_to_img
    from PIL import Image
    from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose
    gamma = 0.95
    action_space = 5
    num_obj = 12
    action_size = action_space * num_obj
    save_path = 'simple_MCTS_inherit_100_12obj_transpose_shape'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open('test35.pkl','rb') as fp:
        test_configs = pickle.load(fp)

    # env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose(size=(35,35))
    env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape(size=(35,35))
    exp = num_obj

    
    for idx, test_config in enumerate(test_configs):
        print('case',idx)
        frames = []
        pos_,target_,size_ = deepcopy(test_config)
        # print(len(pos_))
        shape = []
        for s in size_:
            ps = []
            for i in range(s[0]):
                for j in range(s[1]):
                    ps.append([i,j])
            shape.append(ps)
        env_test.setmap(pos_,target_,shape, np.zeros(len(pos_)), np.zeros(len(pos_)))
        finished_test = False
        s = 0
        total_reward = 0
        input_map = deepcopy(env_test.getmap())
        target_map = env_test.gettargetmap()

        img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
        frames.append(img)
        tree = MCT()
        tree.root.state = deepcopy(env_test)
        tree.setactionsize(action_size)
        step = 1
        while (not (finished_test == 1)) and step < 100:
            step += 1
            max_depth = 0
            for i in range(action_size):

                if not tree.haschild(tree.root.id, i):
                    env_copy = deepcopy(env_test)
                    item = int(i / action_space)
                    direction = i % action_space
                    reward, done = env_copy.move(item, direction)

                    succ, child_id = tree.expansion(tree.root.id, i, reward, env_copy, done)
                    
                    if succ:

                        reward_sum = 0
                        cnt = 0
                        last_item = -1
                        end_flag = False
                        env_sim = deepcopy(env_copy)
                        while (not (end_flag == 1)) and cnt < 20:
                            cnt += 1
                            jumps = []
                            for i in range(exp):
                                jumps.append((-size_[i][0]*size_[i][1],i))
                        
                            sorted(jumps)
                            
                            cmax = -1
                            item = -1
                            direction = 4

                            for i in jumps:
                                reward, done = env_sim.move(i[1],4)
                                if done != -1:
                                    item = i[1]
                                    break


                            if item == -1:
                                while True:
                                    item = np.random.randint(exp)
                                    if item == last_item:
                                        continue
                                    direction = np.random.randint(4)
                                    
                                    reward, done = env_sim.move(item, direction)
                                    if done != -1:
                                        break

                            reward_sum += reward * pow(gamma, cnt-1)
                            end_flag = done
                                
                        tree.nodedict[child_id].value = reward_sum
                        tree.backpropagation(child_id)
            
            for i in range(100):
                _id = tree.selection()
                action = 0
                cnt = 0
                error = True
                while True:
                    action = np.random.randint(action_size)
                    cnt += 1
                    if cnt > 100:
                        error = True
                        for j in range(action_size):
                            if tree.haschild(_id, j):
                                error = False
                        if error:
                            break

                    if not tree.haschild(_id, action):
                        error = False
                        break

                if error:
                    print('expansion remove error')
                    continue
                                    
                item = int(action / action_space)
                direction = action % action_space
                env_copy = deepcopy(tree.getstate(_id))
                reward, done = env_copy.move(item, direction)

                succ, child_id = tree.expansion(_id, action, reward, env_copy, done)

                if succ:
                    reward_sum = 0
                    cnt = 0
                    last_item = -1
                    end_flag = False
                    env_sim = deepcopy(env_copy)
                    while (not (end_flag == 1)) and cnt < 20:
                        cnt += 1
                        jumps = []
                        for i in range(exp):
                            jumps.append((-size_[i][0]*size_[i][1],i))
                    
                        sorted(jumps)
                        
                        cmax = -1
                        item = -1
                        direction = 4

                        for i in jumps:
                            reward, done = env_sim.move(i[1],4)
                            if done != -1:
                                item = i[1]
                                break


                        if item == -1:
                            while True:
                                item = np.random.randint(exp)
                                if item == last_item:
                                    continue
                                direction = np.random.randint(4)
                                
                                reward, done = env_sim.move(item, direction)
                                if done != -1:
                                    break

                        reward_sum += reward * pow(gamma, cnt-1)
                        end_flag = done

                    tree.nodedict[child_id].value = reward_sum
                    max_depth = max([max_depth, tree.nodedict[child_id].depth])
                    tree.backpropagation(child_id)

            action = tree.root.best
            item = int(action / action_space)
            direction = action % action_space
            reward, done = env_test.move(item, direction)
            finished_test = done
            total_reward += reward
            print('step',step,'times',tree.nodedict[tree.root.childs[action]].times, 'max_depth',max_depth-tree.root.depth)
            tree.nextstep(action)

            if direction == 4:
                route = env_test.getlastroute()
                route_map = np.zeros_like(input_map)
                for node in route:
                    x,y,s,ds = node
                    route_map[x,y] = 1
                
                for node in route:
                    x,y,s,ds = node
                    imap = env_test.getcleanmap(item)
                    ps,bx = env_test.getitem(item, s)
                    for p in ps:
                        xx,yy = p
                        xx += x
                        yy += y
                        imap[xx,yy] = item + 2
                    
                    img = convert_to_img(imap, target_map, route_map)
                    frames.append(img)
                
            
            input_map = deepcopy(env_test.getmap())
            img = convert_to_img(input_map, target_map, np.zeros_like(input_map))
            frames.append(img)
        
        for j,img in enumerate(frames):
            im = Image.fromarray(img)
            im.save('%s/case%d_%d.png'%(save_path,idx,j))

        if finished_test:
            print('Finished! Reward:',total_reward,'Steps:',step)
        else:
            print('Failed Reward:',total_reward,'Steps:',step)
        
        break

def test_naive_agent():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    from MCTS import MCT
    import imageio
    save_path = 'NN_MCTS_50NN_50random_sim_discount'
    gamma = 0.95
    
    exps = [13]
    # illist = [30, 46, 73, 109, 116, 130, 132, 138, 186]
    illist = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        HyperParameters
    '''
    for exp in exps:
        steps = []
        times = []
        rewards_cases = []
        with open('exp35_%d.pkl'%exp,'rb') as fp:
            test_configs = pickle.load(fp)
        
        case_num = len(test_configs)
        # learning_rate = 0.0002
        learning_rate = 0.0002
        total_episodes = 500000000         # Total episodes for training
        max_steps = 100              # Max possible steps in an episode
        batch_size = 64             
        
        frame_num = 1
        obj_num = 17
        action_space = 5
        action_size = 17 * action_space

        state_size = [35,35,(2*17+1)*frame_num]

        weight_paths = []

        # weight_paths.append('./weights_20181128_1/model_20000.ckpt')
        # weight_paths.append('./weights_20190103_1/model_36600.ckpt')

        # save_matrix = np.zeros([case_num+1, len(weight_paths)])
        

        '''
            Setup DQN
        '''
        # net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
        
        # saver = tf.train.Saver()

        # env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(15,15))
        env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(35,35),max_num=obj_num)
        # sess = tf.Session(config=config)

        # for idx, path in enumerate(weight_paths):
        #     saver.restore(sess, path)
        #     cnt = 0
        #     print('load',path)
        succ = 0
        count =0
        for idd, test_config in enumerate(test_configs):
            # if idd in illist:
                # continue
            
            print('case',idd)
            # frames = []
            # last_list = []
            pos_,target_,size_ = deepcopy(test_config)
            env_test.setmap(pos_,target_,size_)
            # env_test.randominit_crowded()
            finished_test = False
            total_reward = 0
            s = 0
            # input_map = deepcopy(env_test.getmap())
            # target_map = env_test.gettargetmap()

            # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
            # frames.append(img)

            # tree = MCT()
            # tree.root.state = deepcopy(env_test)
            t = time.time()
            last_item = -1

            while not finished_test and s < 100:
                s += 1
                max_depth = 0
                jumps = []
                for i in range(exp):
                    action = i*action_space+4
                    env_copy = deepcopy(env_test)
                    reward, done = env_copy.move(i, 4)
                    if done != -1:
                        jumps.append(i)
                
                
                cmax = -1
                item = -1
                direction = 4
                for i in jumps:
                    if i == last_item:
                        continue
                    tmp = size_[i][0]*size_[i][1]
                    if tmp > cmax:
                        cmax = tmp
                        item = i

                if item == -1:
                    while True:
                        item = np.random.randint(exp)
                        if item == last_item:
                            continue
                        direction = np.random.randint(4)
                        env_copy = deepcopy(env_test)
                        reward, done = env_copy.move(item,direction)
                        if done != -1:
                            break
                
                reward, done = env_test.move(item, direction)
                finished_test = done
                # print('item',item,'direction',direction,'reward',reward)
                
                total_reward += reward
                last_item = item
            
            times.append(time.time()-t)
            steps.append(s)
            rewards_cases.append(total_reward)

            if finished_test:
                print('Finished! Reward:',total_reward,'Steps:',s)
                count += 1
            else:
                print('Failed Reward:',total_reward,'Steps:',s)
            
            # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
            # for j,img in enumerate(frames):
            #     im = Image.fromarray(img)
            #     im.save('%s/case%d_%d.png'%(save_path,idd,j))

        with open('exp_%d_step_naive.txt'%exp,'w') as fp:
            for st in steps:
                fp.write('%d\n'%st)

        with open('exp_%d_time_naive.txt'%exp,'w') as fp:
            for t in times:
                fp.write('%f\n'%t)

        with open('exp_%d_reward_naive.txt'%exp,'w') as fp:
            for r in rewards_cases:
                fp.write('%d\n'%r)
        
        print('exp',exp,'mean step',np.array(steps).mean(),'mean time',np.array(times).mean(),'succ',count/200,'total',np.array(rewards_cases).mean())

def test_new_action_one_frame_add_action_all_rewards_MCTS_with_naive_agent():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    from MCTS import MCT
    import imageio

    save_path = 'NN_MCTS_50NN_50random_sim_discount'
    gamma = 0.95
    
    exps = [17]
    # illist = [30, 46, 73, 109, 116, 130, 132, 138, 186]
    illist = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        HyperParameters
    '''
    
        
        
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 1
    obj_num = 17
    action_space = 5
    action_size = 17 * action_space

    state_size = [35,35,(2*17+1)*frame_num]

    weight_paths = []

    # weight_paths.append('./weights_20181128_1/model_20000.ckpt')
    weight_paths.append('./weights_20190103_1/model_36600.ckpt')

    # save_matrix = np.zeros([case_num+1, len(weight_paths)])
    

    '''
        Setup DQN
    '''
    # net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    
    # saver = tf.train.Saver()

    # env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(15,15))
    env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(35,35),max_num=obj_num)
    # sess = tf.Session(config=config)
    lock = threading.Lock()

    for idx, path in enumerate(weight_paths):
        # saver.restore(sess, path)
        cnt = 0
        print('load',path)

        for exp in exps:
            steps = []
            times = []
            rewards_cases = []
            with open('exp35_%d.pkl'%exp,'rb') as fp:
                test_configs = pickle.load(fp)
            

            for idd, test_config in enumerate(test_configs):
                if idd in illist:
                    continue
               

                print('case',idd)
                frames = []
                last_list = []
                pos_,target_,size_ = deepcopy(test_config)
                env_test.setmap(pos_,target_,size_)
                # env_test.randominit_crowded()
                finished_test = False
                total_reward = 0
                s = 0
                input_map = deepcopy(env_test.getmap())
                target_map = env_test.gettargetmap()

                # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)

                tree = MCT()
                tree.root.state = deepcopy(env_test)
                t = time.time()


                

                while (not( finished_test == 1)) and s < 100:
                    s += 1
                    max_depth = 0
                    # threads = []
                    # def simulation(child,action,env):
                        
                                # lock.release()

                    for i in range(action_size):
                        # th = threading.Thread(target=simulation,args=(tree.root.id,i,env_test,max_depth))
                        # th.setDaemon(True)
                        # th.start()
                        # threads.append(th)
                        # simulation(tree.root.id,i,env_test)
                        action = i
                        child = tree.root.id

                        if not tree.haschild(child, action):
                            item = int(action/action_space)
                            direction = action % action_space
                            env_copy = deepcopy(env_test)
                            reward, done = env_copy.move(item, direction)

                            # lock.acquire()
                            succ, child_id = tree.expansion(child, action, reward, env_copy, done)
                            # lock.release()

                            if succ:
                                if done != 1:
                                    cnt = 0
                                    reward_sum = 0
                                    env_sim = deepcopy(env_copy)
                                    end_flag = False

                                    last_item = -1
                                    while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for aa in range(exp):
                                            jumps.append((-size_[aa][0]*size_[aa][1],aa))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for j in jumps:
                                            reward, done = env_sim.move(j[1],4)
                                            if done != -1:
                                                item = j[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue
                                                direction = np.random.randint(4)
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    
                                                    
                                                    break
                                        
                                        end_flag = done
                                        
                                        reward_sum += reward * pow(gamma, cnt-1)
                                        if done == 1:
                                            print('wa done!',reward_sum)
                                        

                                    if reward_sum > 20000:
                                        print('error!!!')
                                    # lock.acquire()
                                    tree.nodedict[child_id].value = reward_sum
                                    # print('reward sum',reward_sum,'id',child_id)
                                    # lock.release()

                                # lock.acquire()
                                max_depth = max([max_depth, tree.nodedict[child_id].depth])
                                tree.backpropagation(child_id)

                    # print()
                    # for t in threads:
                        # t.join()
                    
                    # threads = []
                    # for i in range(100):
                    #     _id = tree.selection()
                    #     env_copy = deepcopy(tree.getstate(_id))
                        
                    #     action = np.random.randint(action_size)
                        
                    #     item = int(action / action_space)
                    #     direction = action % action_space
                    #     reward, done = env_copy.move(item, direction)
                        
                    #     succ, child_id = tree.expansion(_id, action, reward, env_copy, done)

                    #     if succ:
                    #         tree.nodedict[child_id].value = 0
                    #         max_depth = max([max_depth, tree.nodedict[child_id].depth])
                    #         tree.backpropagation(child_id)
                    
                    t_random = time.time()
                    for _ in range(100):
                        child = tree.selection()
                        # env_copy = deepcopy(tree.getstate(_id))
                        jumps = []
                        for i in range(exp):
                            jumps.append((-size_[i][0]*size_[i][1],i))
                        
                        sorted(jumps)

                        env_copy= deepcopy(env_test)
                        action = -1
                        for i in jumps:
                            reward, done = env_copy.move(i[1],4)
                            if done != -1:
                                # item = i[1]
                                action = i[1]*action_space+4
                                break

                        if action == -1:
                            action = np.random.randint(action_size)

                        if not tree.haschild(child, action):
                            item = int(action/action_space)
                            direction = action % action_space
                            env_copy = deepcopy(env_test)
                            reward, done = env_copy.move(item, direction)

                            # lock.acquire()
                            succ, child_id = tree.expansion(child, action, reward, env_copy, done)
                            # lock.release()

                            if succ:
                                if done != 1:
                                    cnt = 0
                                    reward_sum = 0
                                    env_sim = deepcopy(env_copy)
                                    end_flag = False

                                    last_item = -1
                                    while (not (end_flag == 1)) and cnt < 20:
                                        cnt += 1
                                        jumps = []
                                        for i in range(exp):
                                            jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                                        sorted(jumps)
                                        
                                        cmax = -1
                                        item = -1
                                        direction = 4

                                        for i in jumps:
                                            reward, done = env_sim.move(i[1],4)
                                            if done != -1:
                                                item = i[1]
                                                break
        

                                        if item == -1:
                                            while True:
                                                item = np.random.randint(exp)
                                                if item == last_item:
                                                    continue
                                                direction = np.random.randint(4)
                                                
                                                reward, done = env_sim.move(item, direction)
                                                if done != -1:
                                                    
                                                    break
                    
                                        reward_sum += reward * pow(gamma, cnt-1)
                                        end_flag = done
                                        if done == 1:
                                            print('wa done!',reward_sum)
                                    
                                    if reward_sum > 20000:
                                        print('error!!!2')
                                    # lock.acquire()
                                    tree.nodedict[child_id].value = reward_sum
                                    # print('reward sum',reward_sum,'id',child_id)
                                    # lock.release()

                                # lock.acquire()
                                max_depth = max([max_depth, tree.nodedict[child_id].depth])
                                tree.backpropagation(child_id)

                        # simulation(_id,action,tree.getstate(_id))
                        # th = threading.Thread(target=simulation,args=(_id,action,tree.getstate(_id)))
                        # th.setDaemon(True)
                        # th.start()
                        # threads.append(th)
                        # item = int(action / action_space)
                        # direction = action % action_space
                        # reward, done = env_copy.move(item, direction)
                        
                        # succ, child_id = tree.expansion(_id, action, reward, env_copy, done)

                        # if succ:
                        #     if done != 1:
                        #         state = env_copy.getstate_1()
                        #         conflict = env_copy.getconflict()
                        #         finishtag = env_copy.getfinished()

                        #         # value = np.max(Qs)
                        #         cnt = 0
                        #         reward_sum = 0
                        #         env_sim = deepcopy(env_copy)
                        #         end_flag = False
                        #         last_item = -1
                        #         while not end_flag and cnt < 20:
                        #             cnt += 1

                        #             jumps = []
                        #             for i in range(exp):
                        #                 jumps.append((-size_[i][0]*size_[i][1],i))
                                    
                        #             sorted(jumps)
                        #             # for i in range(exp):
                        #             #     action = i*action_space+4
                        #             #     env_sim_ = deepcopy(env_sim)
                        #             #     reward, done = env_sim_.move(i, 4)
                        #             #     if done != -1:
                        #             #         jumps.append(i)
                                    
                                    
                        #             cmax = -1
                        #             item = -1
                        #             direction = 4
                        #             # for i in jumps:
                        #             #     if i == last_item:
                        #             #         continue
                        #             #     tmp = size_[i][0]*size_[i][1]
                        #             #     if tmp > cmax:
                        #             #         cmax = tmp
                        #             #         item = i
                        #             for i in jumps:
                        #                 reward, done = env_sim.move(i[1],4)
                        #                 if done != -1:
                        #                     item = i[1]
                        #                     break

                        #             if item == -1:
                        #                 while True:
                        #                     item = np.random.randint(exp)
                        #                     if item == last_item:
                        #                         continue
                        #                     direction = np.random.randint(4)
                        #                     # env_sim_ = deepcopy(env_sim)
                        #                     # reward, done = env_sim_.move(item,direction)
                        #                     reward, done = env_sim.move(item, direction)
                        #                     if done != -1:
                        #                         break
                        #             # state = env_sim.getstate_1()
                        #             # conflict = env_sim.getconflict()
                        #             # finishtag = env_sim.getfinished()
                                    
                        #             # reward, done = env_sim.move(item, direction)
                        #             # end_flag = done
                                        
                                    
                        #             reward_sum += reward * pow(gamma, cnt-1)

                        #         # tree.nodedict[child_id].value = value
                        #         tree.nodedict[child_id].value = reward_sum
                        #         # print('reward_sum',reward_sum)

                        #     max_depth = max([max_depth, tree.nodedict[child_id].depth])
                        #     tree.backpropagation(child_id)

                    t_random = time.time() - t_random

                    # for t in threads:
                    #     t.join()
                    
                    action = tree.root.best
                    item = int(action / action_space)
                    direction = action % action_space
                    reward, done = env_test.move(item, direction)
                    finished_test = done
                    
                    print('times',tree.nodedict[tree.root.childs[action]].times,'id',tree.root.id, 'max_depth',max_depth-tree.root.depth, 'value',tree.root.value,'best depth',tree.getbestdepth(),'random time',t_random)
                    print('last')
                    for tid in last_list:
                        node = tree.nodedict[tid]
                        print('id',node.id,'value',node.value,'reward',node.reward,'best',node.best,'depth',node.depth-tree.root.depth)

                    last_list = []
                    node = tree.root
                    best = node.best
                    print('this')
                    while best != -1:
                        node = tree.nodedict[node.childs[best]]
                        best = node.best
                        print('id',node.id,'value',node.value,'reward',node.reward,'best',best,'depth',node.depth-tree.root.depth)
                        last_list.append(node.id)
                        

                    tree.nextstep(action)
                    total_reward += reward
                    
                    # if direction == 4:
                    #     route = env_test.getlastroute()
                    #     route_map = np.zeros_like(input_map)
                    #     for node in route:
                    #         x,y = node
                    #         route_map[x,y] = 1
                        
                    #     img = convert_to_img(input_map,target_map,route_map)
                    #     frames.append(img)
                    
                    
                    # input_map = deepcopy(env_test.getmap())
                    # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                    # frames.append(img)
                
                times.append(time.time()-t)
                steps.append(s)
                rewards_cases.append(total_reward)

                if finished_test == 1:
                    print('Finished! Reward:',total_reward,'Steps:',s)
                else:
                    print('Failed Reward:',total_reward,'Steps:',s)
                
                # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
                # for j,img in enumerate(frames):
                #     im = Image.fromarray(img)
                #     im.save('%s/case%d_%d.png'%(save_path,idd,j))

            with open('exp_%d_step_naive_MCTS.txt'%exp,'w') as fp:
                for st in steps:
                    fp.write('%d\n'%st)
            with open('exp_%d_time_naive_MCTS.txt'%exp,'w') as fp:
                for t in times:
                    fp.write('%f\n'%t)
            with open('exp_%d_reward_naive_MCTS.txt'%exp,'w') as fp:
                for r in rewards_cases:
                    fp.write('%d\n'%r)

            print('exp',exp,'mean step',np.array(steps).mean(),'mean time',np.array(times).mean())


def evaluate_nn():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0            # exploration probability at start 
    explore_stop = 0.1            # minimum exploration probability 
    decay_rate = 0.0001            # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate
    map_size = 35
    # frame_num = 5
    obj_num = 17
    action_space = 5
    start_to_train = 256
    state_size = [map_size, map_size,(2*obj_num+1)]
    action_size = obj_num * action_space
    
    test_configs_set = []
    item_num_list = [5,9,13,17]
    
    with open('./exp35_9.pkl','rb') as fp:
        test_configs = pickle.load(fp)
        test_configs_set.append(test_configs)

    current_num_test = 9
    
    '''
        Setup DQN
    '''
    net = DQNetwork12(state_size=state_size,learning_rate=learning_rate,num_objects=obj_num, action_space=action_space)
    


    '''
        Setup env
    '''
    # env = ENV()
    # env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)

    
    saver = tf.train.Saver(max_to_keep=100)
    sess = tf.Session(config=config)
    saver.restore(sess,'./weights_20190103_1/model_36600.ckpt')

    
    env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(map_size,map_size),max_num=obj_num)
    
    total_time = 0
    count = 0
    steps = 0
    sum_reward = []
    for i,test_config in enumerate(test_configs):
        if i == 20:
            break
        pos_,target_,size_ = deepcopy(test_config)
        env_test.setmap(pos_,target_,size_)
        # state_list_test = []
        # action_list_test = []
        finished_test = False
        total_reward = 0
        s = 0
        t = time.time()
        while not finished_test and s < 100:
            s += 1
            state = env_test.getstate_1()
            state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

            conflict_matrix = env_test.getconflict()
            conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
            finish_tag = env_test.getfinished()
            finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")
            finish_tag = np.zeros_like(finish_tag)

            Qs = sess.run(net.output, feed_dict={net.inputs_: state.reshape((1, *state.shape)), net.conflict_matrix: conflict_matrix.reshape((1,*conflict_matrix.shape)), net.finish_tag: finish_tag.reshape((1, *finish_tag.shape))})

            Qs = Qs.squeeze()

            while True:
                action = np.argmax(Qs)
                choice_index = int(action / action_space)
                choice_action = action % action_space
                reward, done = env_test.move(choice_index, choice_action)
                # total_reward += reward
                if done == -1:
                    Qs[action] = -1000000000
                    continue
                
                finished_test = done
                break
            if reward > -50:
                total_reward += reward
            action_index = action
            action = np.zeros(action_space * obj_num)
            action[action_index] = 1
            # action_list_test.append(action)
        total_time += time.time() - t
        if finished_test:
            count += 1
            steps += s
        # else:
            # this_case = env_test.getconfig()
            # failure_buffer.add(this_case)

        sum_reward.append(total_reward)
    
    # sum_reward /= 100
    sum_ = np.mean(sum_reward)
    median_ = np.median(sum_reward)
    # count /= 
    
    total_time /= 20
    print('mean step',steps/count,'mean reward',sum_,'rate',count/20,'time',total_time)
            



def test_new_action_5_frames_add_action_all_rewards_logs():
    from visulization import convert
    from visulization import convert_to_img
    from PIL import Image
    import imageio
    save_path = 'test_add_action_frames_all_rewards_1_frames'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000         # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 64             
    
    frame_num = 5
    obj_num = 5
    action_space = 5

    state_size = [15,15,(2*obj_num+1)*frame_num]


    '''
        Setup DQN
    '''
    net = DQNetwork5(state_size=state_size,learning_rate=learning_rate,action_size=obj_num*action_space, frame_num=frame_num)
    
    saver = tf.train.Saver()

    env_test = ENV_scene_new_action_pre_state_penalty(size=(15,15))
    sess = tf.Session(config=config)
    # saver.restore(sess,'./weights_20181022_1/model_7720000.ckpt')
    # saver.restore(sess,'./weights_20181029_2/model_273000.ckpt')
    saver.restore(sess,'./weights_20181029_2/model_341000.ckpt')

    with open('%s/groundtruth.txt'%(save_path),'w') as fp:
        for i in range(20):
            env_test.randominit_crowded()
            state_list_test = []
            action_list_test = []
            finished_test = False
            total_reward = 0
            s = 0
            frames = []
            input_map = env_test.getmap()
            target_map = env_test.gettargetmap()

            img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
            # frames.append(img)
            im = Image.fromarray(img)
            im.save('%s/%d_0_init.png'%(save_path,i))


            while s < 30:
                s += 1
                state = env_test.getstate_1()

                input_map = np.array(env_test.getmap())
                img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                # frames.append(img)
                im = Image.fromarray(img)
                im.save('%s/%d_%d_init.png'%(save_path,i,s))
                if finished_test:
                    break

                state_list_test.append(state)
                

                if len(state_list_test) < frame_num:
                    dif = frame_num - len(state_list_test)
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros_like(state))

                    cur_two_frame = np.concatenate([*tocon,*state_list_test], -1)
                        
                else:
                    cur_two_frame = np.concatenate(state_list_test[-frame_num:],-1)


                if len(action_list_test) < frame_num-1:
                    dif = frame_num - len(action_list_test) - 1
                    tocon = []
                    for j in range(dif):
                        tocon.append(np.zeros(action_space * obj_num))
                    
                    cur_action_chain = np.concatenate([*tocon,*action_list_test],-1)
                else:
                    cur_action_chain = np.concatenate(action_list_test[-(frame_num-1):],-1)

                Qs = sess.run(net.output, feed_dict={net.inputs_: cur_two_frame.reshape((1, *cur_two_frame.shape)), net.action_chain: cur_action_chain.reshape((1, *cur_action_chain.shape))})

                Qs = Qs.squeeze()
                for ia,_ in enumerate(Qs):
                    env_copy = deepcopy(env_test)
                    choice_index = int(ia/action_space)
                    choice_action = ia % action_space
                    reward, done = env_copy.move(choice_index, choice_action)
                    fp.write('case %d step %d action %d reward %d predict %.3f\n'%(i,s,ia,reward,_))

                while True:
                    action = np.argmax(Qs)
                    choice_index = int(action / action_space)
                    choice_action = action % action_space
                    reward, done = env_test.move(choice_index, choice_action)
                    total_reward += reward
                    if done == -1:
                        Qs[action] = -1000000000
                        continue
                    
                    finished_test = done
                    break

                action_index = action
                action = np.zeros(action_space * obj_num)
                action[action_index] = 1
                action_list_test.append(action)

                choice_action = action_index % action_space
                if choice_action == 4:
                    route = env_test.getlastroute()
                    route_map = np.zeros_like(input_map)
                    for node in route:
                        x,y = node
                        route_map[x,y] = 1
                    
                    img = convert_to_img(input_map,target_map,route_map)
                    frames.append(img)
                
                input_map = env_test.getmap()
                img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
                frames.append(img)
            
            # imageio.mimsave('%s/%d.gif'%(save_path,i), frames, 'GIF', duration = 0.5)
            # for j,img in enumerate(frames):
            #     im = Image.fromarray(img)
            #     im.save('%s/case%d_%d.png'%(save_path,i,j))

def get_fail():
    fail = [0, 4, 7, 9, 23, 24, 26, 30, 31, 33, 36, 42, 46, 48, 49, 53, 60, 62, 68, 69, 70, 73, 85, 107, 109, 116, 124, 130, 132, 138, 139, 146, 153, 154, 165, 179, 184, 185, 186, 187, 197]
    total_fail = [7, 9, 24, 26, 30, 33, 36, 42, 46, 49, 53, 60, 62, 69, 70, 73, 85, 107, 109, 116, 124, 130, 132, 138, 139, 154, 179, 184, 185, 186, 187, 197]
    # path = './test_1110_3_252'
    # # files = [file for file in os.listdir(path)]
    # filenames = []
    # for _ in fail:
    #     filename = 'case%d'%_
    #     files = [file_ for file_ in os.listdir(path) if filename in file_]
    #     for f in files:
    #         print(f)
    #         filenames.append(f)

    # for file_ in filenames:
    #     with open('%s/%s'%(path,file_),'rb') as fp:
    #         with open('dest/%s'%file_,'wb') as dp:
    #             dp.write(fp.read())
    # with open('./evaluation_100steps.txt','r') as fp:
    #     txt = fp.readline()
    #     cnt = 0
    #     fail_list = []
    #     while txt != '':
    #         txt = int(txt.split()[-1])
    #         if txt == 0:
    #             fail_list.append(cnt)
    #         cnt += 1
    #         txt = fp.readline()
    with open('./evaluation_100steps_conflict.txt','r') as fp:
        txt = fp.readline()
        cnt = 0
        fail_list = []
        while txt != '':
            txt = int(txt.split()[-1])
            if txt == 0:
                fail_list.append(cnt)
            cnt += 1
            txt = fp.readline()
    _list = []
    for f in fail_list:
        if f in fail:
            _list.append(f)
    
    print(_list)
    # print(fail_list)


if __name__ == '__main__':
    # get_fail()
    # train()
    # train2()
    # train3()
    # train4()
    # test_c3()
    # test_c5()
    # train_two_stage_task()
    # train_CNN()
    # test_scene()
    # train_comb()
    # train_comb_new_action()
    # train_comb_five_frame()
    # test_new_action_5_frames()
    # test_new_action_5_frames_success_rate()
    # train_comb_five_frame_add_action()
    # train_comb_five_frame_add_action_all_reward()
    # test_new_action_5_frames_add_action_all_rewards()
    # test_new_action_5_frames_add_action_all_rewards_conflict()
    # train_comb_five_frame_add_action_all_reward_loss_details()
    # train_comb_five_frame_add_action_all_reward_loss_details_failure_cases_reinforce()
    # train_comb_five_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_no_conflict()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_no_finish_tag()
    
    # test_simple_MCTS()

    # test_new_action_one_frame_add_action_all_rewards_MCTS()
    # test_naive_agent()
    # test_simple_MCTS_transpose()
    # test_new_action_one_frame_add_action_all_rewards_MCTS_with_naive_agent()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_12()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_12_memory_saving()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_12_memory_saving_random_index()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index_NN11()
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_17_memory_saving_random_index_NN12()
    train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_25_memory_saving_random_index_NN12()
    
    # test_simple_MCTS_transpose()
    # evaluate_nn()

    # test_new_action_5_frames_add_action_all_rewards_logs()
    # train_comb_ten_frame_add_action()

    # test_new_action()
    # test_comb()
    # train_comb_two_frame()
    # test3()
    # from visulization import convert
    # import numpy as np
    # from PIL import Image
    # m = np.zeros([5,5])
    # m[0,0] = 4
    # img = convert(m)
    # img = Image.fromarray(np.uint8(img))
    # img.save('what.png')