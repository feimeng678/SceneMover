import tensorflow as tf
import numpy as np
import os
from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly
from copy import deepcopy
from baselines.common.distributions import make_pdtype
from baselines.a2c.utils import cat_entropy
import tensorflow.keras.losses as tkl
from gym import spaces
from collections import deque
import pickle


class MemoryUp():
    def __init__(self, path='./buffer', bunch=100, max_bunch=200):
        if not os.path.exists(path):
            os.makedirs(path)

        self.buffer = deque(maxlen=1000)
        # self.file_list = [file_ for file_ in os.listdir(path) if file_[-3:] = 'pkl']
        self.num = len([file_ for file_ in os.listdir(path) if file_[-3:] == 'pkl'])
        self.bunch = bunch
        self.path = path
        self.max_bunch = max_bunch
        self.current = self.num

    def add(self, experience):
        self.buffer.append(experience)
        if self.size() > self.bunch:
            data = []
            for i in range(self.bunch):
                data.append(self.buffer.popleft())

            with open(os.path.join(self.path, 'data_%d.pkl' % self.current), 'wb') as fp:
                pickle.dump(data, fp)
            print('data_%d.pkl' % self.current)
            self.current = (self.current + 1) % self.max_bunch
            self.num = max(self.num, self.current)

    def sample(self, batch_size):
        # buffer_size = len(self.buffer)
        # index = np.random.choice(buffer_size,
        #                         size = batch_size,
        #                         replace = False)

        # return [self.buffer[i] for i in index]
        res = []
        index = np.random.choice(self.num,
                                 size=batch_size)

        for i in index:
            with open(os.path.join(self.path, 'data_%d.pkl' % i), 'rb') as fp:
                data = pickle.load(fp)
            id_ = np.random.choice(self.bunch)
            res.append(data[id_])

        return res

    def size(self):
        return len(self.buffer)


class A2C:
    def __init__(self, batch_size, state_size=[64, 64, 2], action_space=5, num_objects=25,lr=0.0002,
                 seq_len=8,gamma = 0.9, name='A2C'):
        self.state_size = state_size
        self.action_size = action_space * num_objects
        self.lr = lr
        self.seq_len = seq_len
        self.gamma = gamma
        self.pdtype = make_pdtype(spaces.Discrete(self.action_size))


        #with tf.variable_scope(name):
        with tf.name_scope(name), tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # We create the placeholders
            # *state_size means that we take each elements of stsize in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.int32, [None, self.seq_len,1], name="actions_")
            self.advantages_ = tf.placeholder(tf.float32, [None, self.seq_len,1], name="advantages_")
            self.rewards_ = tf.placeholder(tf.float32, [None, self.seq_len,1], name="rewards_")
            self.finish_tag = tf.placeholder(tf.float32, [None, self.seq_len, num_objects], name="finish_tag")

            # mask
            self.lr = tf.placeholder(tf.float32, name="learnig_rate")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])  # combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                          filters=64,
                                          kernel_size=[5, 5],
                                          strides=[2, 2],
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1_")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out', self.conv1_out)

            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs=self.conv1_out,
                                            filters=64,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_1")

            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                                   training=True,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")

            self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_out_1,
                                            filters=64,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_2")

            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                                   training=True,
                                                                   epsilon=1e-5,
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
            ## --> [4, 4, 128]
            print('conv2_out', self.conv2_out_2)

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out_2,
                                          filters=128,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out', self.conv3_out)

            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs=self.conv3_out,
                                            filters=128,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_1")

            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                                   training=True,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")

            self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_out_1,
                                            filters=128,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_2")

            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                                   training=True,
                                                                   epsilon=1e-5,
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
            print('conv4_out', self.conv4_out_2)

            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs=self.conv4_out_2,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv5")

            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm5')

            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out', self.conv5_out)

            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs=self.conv5_out,
                                            filters=256,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_1")

            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                                   training=True,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")

            self.conv6_2 = tf.layers.conv2d(inputs=self.conv6_out_1,
                                            filters=256,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_2")

            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                                   training=True,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm6_2')


            #SE3
            self.se3_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv6_batchnorm_2)
            self.se3_2 = tf.keras.layers.Dense(self.se3_1.shape[1]//2)(self.se3_1)
            self.se3_3 = tf.keras.layers.Activation('relu')(self.se3_2)
            self.se3_4 = tf.keras.layers.Dense(self.se3_1.shape[1])(self.se3_3)
            self.se3_5 = tf.keras.layers.Activation('sigmoid')(self.se3_4)
            self.se3_6 = tf.keras.layers.Reshape((1, 1, self.se3_1.shape[1]))(self.se3_5)
            self.conv6_batchnorm_2_se = self.conv6_batchnorm_2 * self.se3_6

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2_se + self.conv5_out, name="conv6_out_2")
            print('conv6_out', self.conv6_out_2)

            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            self.flatten = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            self.flatten = tf.reshape(self.flatten, [-1, self.seq_len, int(self.flatten.shape[-1])])
            print('flatten', self.flatten)

            def lstm_layer(lstm_size, number_of_layers, batch_size):
                '''
                This method is used to create LSTM layer/s for PixelRNN

                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network

                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''

                def cell(size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

                cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, init_state = lstm_layer(256, 2, batch_size)
            outputs, states = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=init_state)
            print(outputs)
            self.rnn = tf.reshape(outputs, [-1, 256])

            self.output_ = tf.layers.dense(inputs=self.rnn,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=None,
                                           name="pre_output_internal")

            #action_branch
            self.output_1 = tf.layers.dense(inputs=self.output_,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=tf.nn.relu,
                                           name="output_1")

            self.pre_act_out = tf.layers.dense(inputs=self.output_1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=self.action_size,
                                           activation=tf.nn.softmax,
                                           name="act_output_internal")
            self.pd, self.pi = self.pdtype.pdfromlatent(self.pre_act_out, init_scale=0.01)

            self.act_out = tf.reshape(self.pi, [-1, self.seq_len, self.action_size], name="act_output_external")

            self.spaction = self.pd.sample()

            #value_branch
            self.output_2 = tf.layers.dense(inputs=self.output_,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=tf.nn.relu,
                                            name="output_2")

            self.output_3 = tf.layers.dense(inputs=self.output_2,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=self.action_size,
                                           activation=tf.nn.relu,
                                            name="output_3")


            self.pre_val_out = tf.layers.dense(inputs=self.output_3,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=1,
                                           activation=tf.nn.tanh,
                                           name="val_output_internal")


            self.val_out = tf.reshape(self.pre_val_out, [-1, self.seq_len, 1], name="val_output_external")

            print("act_out: ",self.act_out)
            print("val_out: ",self.val_out)

            neglogpac = -tf.reduce_sum(tf.log(self.act_out)*tf.one_hot(self.actions_, self.action_size), axis=-1)
            #neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.act_out, labels=tf.one_hot(self.actions_, self.action_size)
            pg_loss = tf.reduce_mean(self.advantages_ * neglogpac)

            vf_loss = tf.reduce_mean(tkl.mean_squared_error(tf.squeeze(self.val_out), self.rewards_))
            entropy = tf.reduce_mean(self.pd.entropy())
            self.loss = pg_loss - entropy * 1e-4 + vf_loss * 0.5

            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


class A2C_evl:
    def __init__(self, batch_size, state_size=[5, 5, 4], action_space=5, num_objects=5, lr=0.0002,
                 seq_len=50, name='A2C'):
        self.state_size = state_size
        self.action_size = action_space * num_objects
        self.learning_rate = lr
        self.seq_len = seq_len
        self.pdtype = make_pdtype(spaces.Discrete(self.action_size))


        with tf.name_scope(name), tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # We create the placeholders
            self.inputs_ = tf.placeholder(tf.float32, [None, self.seq_len, *state_size], name="inputs")
            # self.action_chain = tf.placeholder(tf.float32, [None, self.action_size * (frame_num-1)], name="action_chain")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # self.conflict_matrix = tf.placeholder(tf.float32, [None, num_objects, num_objects, 2], name="conflict_matrix")
            self.finish_tag = tf.placeholder(tf.float32, [None, self.seq_len, num_objects], name="finish_tag")
            # conflict_matrix_and = tf.logical_and(tf.cast(self.conflict_matrix[...,0],tf.bool),tf.cast(self.conflict_matrix[...,1],tf.bool))
            # self.conflict_matrix = tf.cast(self.conflict_matrix,tf.float32)
            # conflict_matrix_and = tf.cast(conflict_matrix_and,tf.float32)
            # self.state_in = ((tf.placeholder(tf.float32, [None, 256], name = "state_in_c1"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h1")),
            #                (tf.placeholder(tf.float32, [None, 256], name = "state_in_c2"), tf.placeholder(tf.float32, [None, 256], name = "state_in_h2")))
            self.state_in = (tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name="lstm_c1"),
                                                           tf.placeholder(tf.float32, [None, 256], name="lstm_h1")),
                             tf.nn.rnn_cell.LSTMStateTuple(tf.placeholder(tf.float32, [None, 256], name="lstm_c2"),
                                                           tf.placeholder(tf.float32, [None, 256], name="lstm_h2")))
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.inputs = tf.reshape(self.inputs_, [-1, *self.state_size])  # combine the first two dims
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                          filters=64,
                                          kernel_size=[5, 5],
                                          strides=[2, 2],
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=False,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [8, 8, 64]
            print('conv1_out', self.conv1_out)

            """
            Second convnet:
            ResNet block
            BatchNormalization 
            ELU
            """
            self.conv2_1 = tf.layers.conv2d(inputs=self.conv1_out,
                                            filters=64,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_1")

            self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1,
                                                                   training=False,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm2_1')

            self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")

            self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_out_1,
                                            filters=64,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv2_2")

            self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2,
                                                                   training=False,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm2_2')

            self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2 + self.conv1_out, name="conv2_out_2")
            ## --> [4, 4, 128]
            print('conv2_out', self.conv2_out_2)

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out_2,
                                          filters=128,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=False,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            print('conv3_out', self.conv3_out)

            """
            Forth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv4_1 = tf.layers.conv2d(inputs=self.conv3_out,
                                            filters=128,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_1")

            self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1,
                                                                   training=False,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm4_1')

            self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")

            self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_out_1,
                                            filters=128,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv4_2")

            self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2,
                                                                   training=False,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm4_2')

            self.conv4_out_2 = tf.nn.elu(self.conv4_batchnorm_2 + self.conv3_out, name="conv4_out_2")
            print('conv4_out', self.conv4_out_2)
            ## --> [4, 4, 128]

            """
            Fifth convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 15*15*55
            self.conv5 = tf.layers.conv2d(inputs=self.conv4_out_2,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="SAME",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv5")

            self.conv5_batchnorm = tf.layers.batch_normalization(self.conv5,
                                                                 training=False,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm5')

            self.conv5_out = tf.nn.elu(self.conv5_batchnorm, name="conv5_out")
            print('conv5_out', self.conv5_out)

            """
            Sixth convnet:
            ResNet block
            BatchNormalization
            ELU
            """
            self.conv6_1 = tf.layers.conv2d(inputs=self.conv5_out,
                                            filters=256,
                                            kernel_size=[3, 3],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_1")

            self.conv6_batchnorm_1 = tf.layers.batch_normalization(self.conv6_1,
                                                                   training=False,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm6_1')

            self.conv6_out_1 = tf.nn.elu(self.conv6_batchnorm_1, name="conv6_out_1")

            self.conv6_2 = tf.layers.conv2d(inputs=self.conv6_out_1,
                                            filters=256,
                                            kernel_size=[1, 1],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name="conv6_2")

            self.conv6_batchnorm_2 = tf.layers.batch_normalization(self.conv6_2,
                                                                   training=False,
                                                                   epsilon=1e-5,
                                                                   name='batch_norm6_2')

            self.conv6_out_2 = tf.nn.elu(self.conv6_batchnorm_2 + self.conv5_out, name="conv6_out_2")
            print('conv6_out', self.conv6_out_2)

            self.finish_tag_ = tf.reshape(self.finish_tag, [-1, num_objects])
            # print("finish_tag")
            # print(self.finish_tag_.shape)
            self.flatten_ = tf.concat([tf.contrib.layers.flatten(self.conv6_out_2), self.finish_tag_], -1)
            # print("flatten_")
            # print(self.flatten_.shape)
            self.flatten = tf.reshape(self.flatten_, [-1, self.seq_len, int(self.flatten_.shape[-1])])

            # print("flatten")
            # print(self.flatten.shape)
            ## --> [1152]

            def lstm_layer(lstm_size, number_of_layers):
                '''
                This method is used to create LSTM layer/s for PixelRNN

                Input(s): lstm_cell_unitis - used to define the number of units in a LSTM layer
                    number_of_layers - used to define how many of LSTM layers do we want in the network
                    batch_size - in this method this information is used to build starting state for the network

                Output(s): cell - lstm layer
                    init_state - zero vectors used as a starting state for the network
                '''

                def cell(size):
                    return tf.contrib.rnn.BasicLSTMCell(lstm_size)

                cell = tf.contrib.rnn.MultiRNNCell([cell(lstm_size) for _ in range(number_of_layers)])

                init_state = cell.zero_state(batch_size, tf.float32)

                return cell, init_state

            cell, self.init_state = lstm_layer(256, 2)
            self.rnn, self.state_out = tf.nn.dynamic_rnn(cell, self.flatten, initial_state=self.state_in)
            print(self.rnn)

            self.output_ = tf.layers.dense(inputs=self.rnn,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=self.action_size,
                                           activation=None,
                                           name="output_internal")
            #action_branch
            self.output_1 = tf.layers.dense(inputs=self.output_,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=tf.nn.relu)

            self.pre_act_out = tf.layers.dense(inputs=self.output_1,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=self.action_size,
                                           activation=tf.nn.softmax,
                                           name="act_output_internal")
            self.pd, self.pi = self.pdtype.pdfromlatent(self.pre_act_out, init_scale=0.01)

            self.act_out = tf.reshape(self.pi, [-1, self.seq_len, self.action_size], name="act_output_external")

            self.spaction = self.pd.sample()

            #value_branch
            self.output_2 = tf.layers.dense(inputs=self.output_,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=256,
                                           activation=tf.nn.relu)

            self.output_3 = tf.layers.dense(inputs=self.output_2,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=self.action_size,
                                           activation=tf.nn.relu)


            self.pre_val_out = tf.layers.dense(inputs=self.output_3,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           units=1,
                                           activation=tf.nn.tanh,
                                           name="val_output_internal")


            self.val_out = tf.reshape(self.pre_val_out, [-1, self.seq_len, 1], name="val_output_external")

            print("act_out: ",self.act_out)
            print("val_out: ",self.val_out)



def trainA2C():

    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000  # Total episodes for training
    max_steps = 100  # Max possible steps in an episode
    batch_size = 1

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.1  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95  # Discounting rate
    map_size = 64
    # frame_num = 5
    obj_num = 25
    action_space = 5
    start_to_train = 100
    seq_len = 8
    state_size = [map_size, map_size, 2]
    action_size = obj_num * action_space
    dim_h = 256
    lr_step = [0, 2000, 4000]
    lrs = [0.0002, 0.00005, 0.00001]

    test_configs_set = []
    item_num_list = [5, 10, 15, 20]

    # import retrain model
    # with open('./exp64_5.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./exp64_10.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./exp64_15.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./exp64_20.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)

    tensorboard_path = "tensorboard/20201021/"
    weight_path = "weights_20201021"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net = A2C_evl(batch_size=batch_size, seq_len=1, state_size=state_size, lr=learning_rate,
                            num_objects=obj_num, action_space=action_space)
    net_train = A2C(batch_size=batch_size, seq_len=1, state_size=state_size, lr=learning_rate,
                            num_objects=obj_num, action_space=action_space)
    # net = A2C_eval(batch_size=1, seq_len=1, state_size=state_size, lr=learning_rate,
    #                        num_objects=obj_num, action_space=action_space)
    # net_infer = A2C_eval(batch_size=batch_size, seq_len=seq_len, state_size=state_size,
    #                              lr=learning_rate, num_objects=obj_num, action_space=action_space)

    '''
        Setup buffer
    '''
    buffer = MemoryUp('./buffer/train', 10)
    failure_buffer = MemoryUp('./buffer/failure', 10, 10)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),
                                                                                         max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net_train.loss)


    write_op = tf.summary.merge_all()

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list, max_to_keep=100)

    current_num = np.random.randint(obj_num) + 1
    env.randominit_crowded(current_num)

    finished = 0
    mx_steps = 0

    current_num = obj_num
    print('start make')

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())


    decay_step = 0
    started_from = 0
    op_step = 0
    failure_sample = 0.4


    h_state = net.init_state

    current_num = np.random.randint(obj_num) + 1
    env.randominit_crowded(current_num)

    finished = 0
    mx_steps = 0

    z_state = np.zeros([1, dim_h])
    z_state_b = np.zeros([batch_size, dim_h])
    h_state = ((z_state, z_state), (z_state, z_state))
    h_state_b = ((z_state_b, z_state_b), (z_state_b, z_state_b))

    optimize_frequency = 1

    state_list = []
    returns = 0
    val_out = 0
    for step in range(started_from, total_episodes):
        # if len(state_list) == seq_len or finished == 1 or mx_steps == max_steps:
        #
        #     mask = np.ones(seq_len)
        #     _ = len(state_list)
        #     for i in range(_, seq_len):
        #         mask[i] = 0
        #         state_list.append(np.zeros_like(state_list[0]))
        #         reward_list.append(np.zeros_like(reward_list[0]))
        #         finish_tag_list.append(np.zeros_like(finish_tag_list[0]))
        #         nex_frame_pack_list.append(np.zeros_like(nex_frame_pack_list[0]))
        #         nex_finish_tag_pack_list.append(np.zeros_like(nex_finish_tag_pack_list[0]))
        #         done_list.append(np.zeros_like(done_list[0]))
        #
        #     buffer.add((mask, state_list, reward_list, finish_tag_list, nex_frame_pack_list, nex_finish_tag_pack_list,
        #                 done_list))
        #
        #     state_list = []
        #     reward_list = []
        #     finish_tag_list = []
        #     nex_frame_pack_list = []
        #     nex_finish_tag_pack_list = []
        #     done_list = []
        #
        # if finished == 1 or mx_steps == max_steps:
        #     current_num = np.random.randint(obj_num) + 1
        #     env.randominit_crowded(current_num)
        #
        #     finished = 0
        #     mx_steps = 0
        #     h_state = ((z_state, z_state), (z_state, z_state))

        state = env.getstate_3()

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        finish_tag = env.getfinished()

        # TODO(me) change this to be a naive policy
        if (explore_probability > exp_exp_tradeoff):
            env_copy = deepcopy(env)

            while True:
                action_index = np.random.randint(action_space * obj_num)
                choice_index = int(action_index / action_space)
                choice_action = action_index % action_space
                reward, done = env_copy.move(choice_index, choice_action)

                if done == -1:
                    continue

                break
            action_out = action_index

        else:
            # TODO fit the input of the network to the lstm
            state_ = env.getstate_3()
            action_out,val_out, h_state = sess.run([net.spaction,net.val_out, net.state_out],
                                   feed_dict={net.inputs_: state_.reshape((1, 1, *state_.shape)),
                                              net.finish_tag: finish_tag.reshape((1, 1, *finish_tag.shape)),
                                              net.state_in: h_state})
            # act_out = act_out.squeeze()
            # Take the biggest Q value (= the best action)

            env_copy = deepcopy(env)
            while True:
                #choice = np.argmax(act_out)
                choice = action_out
                choice_index = int(choice / action_space)
                choice_action = choice % action_space
                reward, done = env_copy.move(choice_index, choice_action)
                break

        action_index = action_out
        choice_index = int(action_index / action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        next_state = env.getstate_3()
        finish_tag = env.getfinished()
        next_action_out, next_val_out, h_state = sess.run([net.spaction, net.val_out, net.state_out],
                                                feed_dict={net.inputs_: next_state.reshape((1, 1, *next_state.shape)),
                                                           net.finish_tag: finish_tag.reshape(
                                                               (1, 1, *finish_tag.shape)),
                                                           net.state_in: h_state})



        returns = reward + gamma * returns * (1 - done)
        advantages = returns - val_out

        # optimize two times
        sess.run(net_train.optimizer, feed_dict={net_train.inputs_: state.reshape((1, 1, *state.shape)), net_train.actions_: np.array([[[action_out]]]),
                                                     net_train.advantages_: np.array([[[advantages]]]),
                                                     net_train.rewards_: np.array([[[returns]]]), net_train.finish_tag: finish_tag.reshape(1, 1, *finish_tag.shape),
                                                     net_train.lr: learning_rate})

        summary = sess.run(write_op, feed_dict={net_train.inputs_: state.reshape((1, 1, *state.shape)), net_train.actions_: np.array([[[action_out]]]),
                                                     net_train.advantages_: np.array([[[advantages]]]),
                                                     net_train.rewards_: np.array([[[returns]]]), net_train.finish_tag: finish_tag.reshape((1, 1, *finish_tag.shape)),
                                                     net_train.lr: learning_rate})

        loss = sess.run(net_train.loss, feed_dict={net_train.inputs_: state.reshape((1, 1, *state.shape)), net_train.actions_: np.array([[[action_out]]]),
                                                     net_train.advantages_: np.array([[[advantages]]]),
                                                     net_train.rewards_: np.array([[[returns]]]), net_train.finish_tag: finish_tag.reshape(1, 1, *finish_tag.shape),
                                                     net_train.lr: learning_rate})

        print(' step:', step, ' reward:', reward, ' done:', done,'net_train_loss: ',loss)

        summt = tf.Summary()
        summt.value.add(tag='learning rate', simple_value=learning_rate)

        writer.add_summary(summary, int(step / optimize_frequency))
        writer.add_summary(summt, int(step / optimize_frequency))

        if int(step / optimize_frequency) % 200 == 0 and step > 0:
                summt = tf.Summary()
                # TODO revise this part
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(
                    size=(map_size, map_size), max_num=obj_num)

                for ii, test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for _ in range(10):
                        test_config = test_configs[_]
                        pos, target, shape, cstate, tstate, wall = deepcopy(test_config)
                        env_test.setmap(pos, target, shape, cstate, tstate, wall)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0

                        h_state_test = ((z_state, z_state), (z_state, z_state))

                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_3()
                            finish_tag = env_test.getfinished()
                            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            act_out, val_out, h_state_test = sess.run([net.act_out, net.val_out, net.state_out],
                                                                    feed_dict={net.inputs_: state.reshape(
                                                                        (1, 1, *state.shape)),
                                                                               net.finish_tag: finish_tag.reshape(
                                                                                   (1, 1, *finish_tag.shape)),
                                                                               net.state_in: h_state_test})

                            act_out = act_out.squeeze()

                            while True:
                                action = np.argmax(act_out)
                                choice_index = int(action / action_space)
                                choice_action = action % action_space
                                reward, done = env_test.move(choice_index, choice_action)
                                total_reward += reward
                                if done == -1:
                                    act_out[action] = -1000000000
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
                    count /= 10
                    summt.value.add(tag='reward_test_%d' % (current_num_test), simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d' % (current_num_test), simple_value=median_)
                    summt.value.add(tag='success rate_%d' % (current_num_test), simple_value=count)

                    writer.add_summary(summt, int(step / optimize_frequency))

        if int(step / optimize_frequency) % 200 == 0 and step > 0:  # !!!!! have been modified!!
                print('model %d saved' % (int(step / optimize_frequency)))
                saver.save(sess, os.path.join(weight_path, 'model_%d.ckpt' % (int(step / optimize_frequency))))

        mx_steps += 1





def train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_25_random_index_NN12_poly_2_channel_net17():
    '''
        HyperParameters
    '''
    # learning_rate = 0.0002
    learning_rate = 0.0002
    total_episodes = 500000000  # Total episodes for training
    max_steps = 100  # Max possible steps in an episode
    batch_size = 4

    # Exploration parameters for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.1  # minimum exploration probability
    decay_rate = 0.0001  # exponential decay rate for exploration prob

    # Q learning hyperparameters
    gamma = 0.95  # Discounting rate
    map_size = 64
    # frame_num = 5
    obj_num = 25
    action_space = 5
    start_to_train = 100
    seq_len = 8
    state_size = [map_size, map_size, 2]
    action_size = obj_num * action_space
    dim_h = 256
    lr_step = [0, 2000, 4000]
    lrs = [0.0002, 0.00005, 0.00001]

    test_configs_set = []
    item_num_list = [5, 10, 15, 20]

    # import retrain model
    # with open('./exp64_5.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./exp64_10.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./exp64_15.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)
    #
    # with open('./exp64_20.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    #     test_configs_set.append(test_configs)

    tensorboard_path = "tensorboard/20190619_1/"
    weight_path = "weights_20190619_1"

    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    '''
        Setup DQN
    '''
    net_train = A2C(batch_size=batch_size, seq_len=seq_len, state_size=state_size, lr=learning_rate,
                            num_objects=obj_num, action_space=action_space)
    net = A2C_eval(batch_size=1, seq_len=1, state_size=state_size, lr=learning_rate,
                           num_objects=obj_num, action_space=action_space)
    net_infer = A2C_eval(batch_size=batch_size, seq_len=seq_len, state_size=state_size,
                                 lr=learning_rate, num_objects=obj_num, action_space=action_space)

    '''
        Setup buffer
    '''
    buffer = MemoryUp('./buffer/train', 10)
    failure_buffer = MemoryUp('./buffer/failure', 10, 10)

    '''
        Setup env
    '''
    # env = ENV()
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),
                                                                                         max_num=obj_num)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter(tensorboard_path)

    ## Losses
    tf.summary.scalar("Loss", net_train.loss)

    # for i in range(action_space):
    #     tf.summary.scalar("Loss of Action %d:" % (i + 1), net_train.loss_details[i])

    write_op = tf.summary.merge_all()

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list, max_to_keep=100)

    state_list = []
    reward_list = []
    finish_tag_list = []
    nex_frame_pack_list = []
    nex_finish_tag_pack_list = []
    done_list = []

    current_num = np.random.randint(obj_num) + 1
    env.randominit_crowded(current_num)

    finished = 0
    mx_steps = 0

    current_num = obj_num
    print('start make')
    for i in range(start_to_train):
        if len(state_list) == seq_len or finished == 1 or mx_steps == max_steps:

            mask = np.ones(seq_len)
            _ = len(state_list)
            print(_, finished)
            for __ in range(_, seq_len):
                mask[__] = 0
                state_list.append(np.zeros_like(state_list[0]))
                reward_list.append(np.zeros_like(reward_list[0]))
                finish_tag_list.append(np.zeros_like(finish_tag_list[0]))
                nex_frame_pack_list.append(np.zeros_like(nex_frame_pack_list[0]))
                nex_finish_tag_pack_list.append(np.zeros_like(nex_finish_tag_pack_list[0]))
                done_list.append(np.zeros_like(done_list[0]))

            if len(done_list) == 0:
                print('error')

            buffer.add((mask, state_list, reward_list, finish_tag_list, nex_frame_pack_list, nex_finish_tag_pack_list,
                        done_list))

            state_list = []
            reward_list = []
            finish_tag_list = []
            nex_frame_pack_list = []
            nex_finish_tag_pack_list = []
            done_list = []

        if finished == 1 or mx_steps == max_steps:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)

            finished = 0
            mx_steps = 0

        state = env.getstate_3()
        state_list.append(state)
        finish_tag_list.append(env.getfinished())

        rewards = []
        dones = []
        nex_frame_pack = []
        nex_finish_tag_pack = []

        for s in range(10):
            env_copy = deepcopy(env)
            action_out,val_out, h_state = sess.run([net.spaction,net.val_out, net.state_out],
                                   feed_dict={net.inputs_: state.reshape((1, 1, *state.shape)),
                                              net.state_in: h_state})
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a / action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            dones.append(done)
            next_state = env_copy.getstate_3()

            finish_tag = env_copy.getfinished()

            nex_finish_tag_pack.append(finish_tag)

            nex_frame_pack.append(next_state)

        nex_finish_tag_pack = np.array(nex_finish_tag_pack)

        reward_list.append(rewards)
        nex_frame_pack_list.append(nex_frame_pack)
        nex_finish_tag_pack_list.append(nex_finish_tag_pack)
        done_list.append(dones)

        print(" i:", i, " next_state_shape:", next_state.shape, " next_finish_tag_pack_shape:",
              nex_finish_tag_pack.shape)

        env_copy = deepcopy(env)
        while True:
            action_index = np.random.randint(action_space * obj_num)
            choice_index = int(action_index / action_space)
            choice_action = action_index % action_space
            reward, done = env_copy.move(choice_index, choice_action)

            if done == -1:
                continue

            break
        # action_index = np.random.randint(obj_num * action_space)
        action = np.zeros(action_space * obj_num)
        action[action_index] = 1

        choice_index = int(action_index / action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)

        finished = done

        mx_steps += 1

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    # saver.restore(sess,'./weights_20190513_1/model_2400.ckpt')
    # saver.restore(sess,'./weights_20190525_1/model_40400.ckpt')

    decay_step = 0

    # started_from = 28801
    # started_from = 40401
    started_from = 0
    op_step = 0
    failure_sample = 0.4

    # data for the time steps
    state_list = []
    reward_list = []
    finish_tag_list = []
    nex_frame_pack_list = []
    nex_finish_tag_pack_list = []
    done_list = []

    h_state = net.init_state

    current_num = np.random.randint(obj_num) + 1
    env.randominit_crowded(current_num)

    finished = 0
    mx_steps = 0

    z_state = np.zeros([1, dim_h])
    z_state_b = np.zeros([batch_size, dim_h])
    h_state = ((z_state, z_state), (z_state, z_state))
    h_state_b = ((z_state_b, z_state_b), (z_state_b, z_state_b))

    for step in range(started_from, total_episodes):
        if len(state_list) == seq_len or finished == 1 or mx_steps == max_steps:

            mask = np.ones(seq_len)
            _ = len(state_list)
            for i in range(_, seq_len):
                mask[i] = 0
                state_list.append(np.zeros_like(state_list[0]))
                reward_list.append(np.zeros_like(reward_list[0]))
                finish_tag_list.append(np.zeros_like(finish_tag_list[0]))
                nex_frame_pack_list.append(np.zeros_like(nex_frame_pack_list[0]))
                nex_finish_tag_pack_list.append(np.zeros_like(nex_finish_tag_pack_list[0]))
                done_list.append(np.zeros_like(done_list[0]))

            buffer.add((mask, state_list, reward_list, finish_tag_list, nex_frame_pack_list, nex_finish_tag_pack_list,
                        done_list))

            state_list = []
            reward_list = []
            finish_tag_list = []
            nex_frame_pack_list = []
            nex_finish_tag_pack_list = []
            done_list = []

        if finished == 1 or mx_steps == max_steps:
            current_num = np.random.randint(obj_num) + 1
            env.randominit_crowded(current_num)

            finished = 0
            mx_steps = 0
            h_state = ((z_state, z_state), (z_state, z_state))

        state = env.getstate_3()
        state_list.append(state)
        finish_tag_list.append(env.getfinished())

        rewards = []
        nex_frame_pack = []
        dones = []
        nex_finish_tag_pack = []

        for a in range(action_size):
            env_copy = deepcopy(env)
            action = np.zeros(action_size)
            action[a] = 1
            choice_index = int(a / action_space)
            choice_action = a % action_space
            reward, done = env_copy.move(choice_index, choice_action)
            rewards.append(reward)
            next_state = env_copy.getstate_3()
            dones.append(done)
            finish_tag = env_copy.getfinished()

            nex_finish_tag_pack.append(finish_tag)
            nex_frame_pack.append(next_state)

        nex_finish_tag_pack = np.array(nex_finish_tag_pack)
        reward_list.append(rewards)
        nex_frame_pack_list.append(nex_frame_pack)
        nex_finish_tag_pack_list.append(nex_finish_tag_pack)
        done_list.append(dones)

        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        finish_tag = env.getfinished()
        # finish_tag = np.pad(finish_tag,(0,obj_num-current_num),"constant")

        # TODO(me) change this to be a naive policy
        if (explore_probability > exp_exp_tradeoff):
            env_copy = deepcopy(env)

            while True:
                action_index = np.random.randint(action_space * obj_num)
                choice_index = int(action_index / action_space)
                choice_action = action_index % action_space
                reward, done = env_copy.move(choice_index, choice_action)

                if done == -1:
                    continue

                break

            action = np.zeros(action_space * obj_num)
            action[action_index] = 1

        else:
            # TODO fit the input of the network to the lstm
            state_ = env.getstate_3()
            action_out,val_out, h_state = sess.run([net.spaction,net.val_out, net.state_out],
                                   feed_dict={net.inputs_: state_.reshape((1, 1, *state_.shape)),
                                              net.finish_tag: finish_tag.reshape((1, 1, *finish_tag.shape)),
                                              net.state_in: h_state})
            # act_out = act_out.squeeze()
            # Take the biggest Q value (= the best action)

            env_copy = deepcopy(env)
            while True:
                #choice = np.argmax(act_out)
                choice = action_out
                choice_index = int(choice / action_space)
                choice_action = choice % action_space
                reward, done = env_copy.move(choice_index, choice_action)

                # if done == -1:
                #     act_out[choice] = -1000000000
                #     continue

                break

            #action = np.zeros(action_space * obj_num)
            #action[choice] = 1

        #action_index = np.argmax(action)
        action_index = action_out
        choice_index = int(action_index / action_space)
        choice_action = action_index % action_space
        reward, done = env.move(choice_index, choice_action)
        print(' step:', step, ' reward:', reward, ' done:', done)

        # buffer.add((mask, state_list, reward_list, finish_tag_list, nex_frame_pack_list, nex_finish_tag_pack_list, done_list))
        finished = done

        optimize_frequency = 12

        if step % optimize_frequency == 0:  # prevent the over sampling.
            decay_step += 1
            print(int(step / optimize_frequency), step)
            batch = buffer.sample(batch_size)
            # print('OK1')
            # [bs, time]
            mask_mb = np.array([each[0] for each in batch])
            # print('mask_mb',mask_mb.shape)

            # [bs, time, a, w, h, c]
            states_mb = np.array([each[1] for each in batch])
            # print('states_mb',states_mb.shape)

            # [bs, time, a]
            rewards_mb = np.array([each[2] for each in batch])
            # print('rewards_mb',rewards_mb.shape)

            # [bs, time, num_obj]
            finish_tag_mb = np.array([each[3] for each in batch])
            # print('finish_tag_mb',finish_tag_mb.shape)

            # [bs, time, a, w, h, c]
            next_states_mb = np.array([each[4] for each in batch])
            # print('next_states_mb',next_states_mb.shape)

            # [bs, time, a, num_obj]
            finish_tag_next_mb = np.array([each[5] for each in batch])
            # print('finish_tag_next_mb',finish_tag_next_mb.shape)

            # [bs, time, a]
            # print([each[6] for each in batch])
            dones_mb = np.array([each[6] for each in batch]).astype(np.float32)
            # print('dones_mb',dones_mb.shape)

            actions_mb = np.ones([batch_size, seq_len, action_size])
            # print(next_states_mb.shape, next_action_chain_mb.shape,conflict_matrix_next_mb.shape)
            target_Qs_batch = []

            for a in range(action_size):
                # TODO fit the input of the network
                # Qs [bs, time, a]
                Qs_next_state = sess.run(net_infer.output, feed_dict={net_infer.inputs_: next_states_mb[:, :, a],
                                                                      net_infer.finish_tag: finish_tag_next_mb[:, :, a],
                                                                      net_infer.state_in: h_state_b})

                Qs = np.max(Qs_next_state, axis=-1)
                target_Qs_batch.append(Qs)

            # [bs, time, a]
            targets_mb = np.array(target_Qs_batch).transpose([1, 2, 0])

            targets_mb = rewards + gamma * dones_mb * targets_mb

            # targets_mb = np.array([each for each in target_Qs_batch]).transpose()
            for it, _ in enumerate(lr_step):
                if decay_step > _:
                    lr = lrs[it]

            # if step < 1000:
            # optimize two times
            sess.run(net_train.optimizer, feed_dict={net_train.inputs_: states_mb, net_train.target_Q_: targets_mb,
                                                     net_train.actions_: actions_mb,
                                                     net_train.finish_tag: finish_tag_mb, net_train.mask: mask_mb,
                                                     net_train.lr: lr})

            summary = sess.run(write_op, feed_dict={net_train.inputs_: states_mb, net_train.target_Q_: targets_mb,
                                                    net_train.actions_: actions_mb, net_train.finish_tag: finish_tag_mb,
                                                    net_train.mask: mask_mb})

            # print("run")
            # else:
            #     _, summary = sess.run([net_train.optimizer2, write_op],feed_dict={net_train.inputs_ : states_mb, net_train.target_Q_ : targets_mb, net_train.actions_: actions_mb, net_train.finish_tag: finish_tag_mb, net_train.mask: mask_mb})
            summt = tf.Summary()
            summt.value.add(tag='learning rate', simple_value=lr)

            writer.add_summary(summary, int(step / optimize_frequency))
            writer.add_summary(summt, int(step / optimize_frequency))

            if int(step / optimize_frequency) % 200 == 0 and step > 0:
                summt = tf.Summary()
                # TODO revise this part
                env_test = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(
                    size=(map_size, map_size), max_num=obj_num)

                for ii, test_configs in enumerate(test_configs_set):
                    current_num_test = item_num_list[ii]
                    sum_reward = []
                    count = 0
                    for _ in range(10):
                        test_config = test_configs[_]
                        pos, target, shape, cstate, tstate, wall = deepcopy(test_config)
                        env_test.setmap(pos, target, shape, cstate, tstate, wall)
                        # state_list_test = []
                        # action_list_test = []
                        finished_test = False
                        total_reward = 0
                        s = 0

                        h_state_test = ((z_state, z_state), (z_state, z_state))

                        while not finished_test and s < 20:
                            s += 1
                            state = env_test.getstate_3()
                            # state = np.pad(state,((0,0),(0,0),(0,(obj_num-current_num_test)*2)),"constant")

                            # conflict_matrix = env_test.getconflict()
                            # conflict_matrix = np.pad(conflict_matrix,((0,obj_num-current_num_test),(0,obj_num-current_num_test),(0,0)),"constant")
                            finish_tag = env_test.getfinished()
                            # finish_tag = np.pad(finish_tag,(0,obj_num-current_num_test),"constant")

                            Qs, h_state_test = sess.run([net.output, net.state_out],
                                                        feed_dict={net.inputs_: state.reshape((1, 1, *state.shape)),
                                                                   net.finish_tag: finish_tag.reshape(
                                                                       (1, 1, *finish_tag.shape)),
                                                                   net.state_in: h_state_test})

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
                    count /= 10
                    summt.value.add(tag='reward_test_%d' % (current_num_test), simple_value=sum_)
                    summt.value.add(tag='reward_test_median_%d' % (current_num_test), simple_value=median_)
                    summt.value.add(tag='success rate_%d' % (current_num_test), simple_value=count)

                    writer.add_summary(summt, int(step / optimize_frequency))

            if int(step / optimize_frequency) % 200 == 0 and step > 0:  # !!!!! have been modified!!
                print('model %d saved' % (int(step / optimize_frequency)))
                saver.save(sess, os.path.join(weight_path, 'model_%d.ckpt' % (int(step / optimize_frequency))))

        mx_steps += 1


if __name__ == '__main__':
    #train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_25_random_index_NN12_poly_2_channel_net17()
    #A2C(1)
    trainA2C()