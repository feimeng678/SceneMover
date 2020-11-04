"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow 1.8.0
gym 0.9.2
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading, queue
from src.env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EP_MAX = 5000
EP_LEN = 100
N_WORKER = 4  # parallel workers
GAMMA = 0.95  # reward discount factor
A_LR = 0.0002  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 16  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
UPDATE_GLOBAL_ITER = 8


map_size = 64
obj_num = 25
action_type = 5

env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),max_num=obj_num)
S_DIM = [map_size, map_size, 2]
flattenS_DIM = map_size * map_size * 2
A_DIM = action_type * obj_num

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tensorboard_path = "tensorboard/20201104/"
weight_path = "weights_20201104/"


if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

if not os.path.exists(weight_path):
    os.makedirs(weight_path)

class PPONet(object):
    def __init__(self):
        self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, *S_DIM], 'state')

        #shared
        w_init = tf.random_normal_initializer(0., .1)
        self.conv1 = tf.layers.conv2d(inputs=self.tfs, filters=64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv1")

        self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1, training=True, epsilon=1e-5,
                                                             name='batch_norm1')

        self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

        self.conv2_1 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                        padding="SAME",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name="conv2_1")

        self.conv2_batchnorm_1 = tf.layers.batch_normalization(self.conv2_1, training=True, epsilon=1e-5,
                                                               name='batch_norm2_1')

        self.conv2_out_1 = tf.nn.elu(self.conv2_batchnorm_1, name="conv2_out_1")

        self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_out_1, filters=64, kernel_size=[1, 1], strides=[1, 1],
                                        padding="SAME",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name="conv2_2")

        self.conv2_batchnorm_2 = tf.layers.batch_normalization(self.conv2_2, training=True, epsilon=1e-5,
                                                               name='batch_norm2_2')

        # SE1
        self.se1_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv2_batchnorm_2)
        self.se1_2 = tf.keras.layers.Dense(self.se1_1.shape[1] // 2)(self.se1_1)
        self.se1_3 = tf.keras.layers.Activation('relu')(self.se1_2)
        self.se1_4 = tf.keras.layers.Dense(self.se1_1.shape[1])(self.se1_3)
        self.se1_5 = tf.keras.layers.Activation('sigmoid')(self.se1_4)
        self.se1_6 = tf.keras.layers.Reshape((1, 1, self.se1_1.shape[1]))(self.se1_5)

        self.conv2_batchnorm_2_se = self.conv2_batchnorm_2 * self.se1_6

        self.conv2_out_2 = tf.nn.elu(self.conv2_batchnorm_2_se + self.conv1_out, name="conv2_out_2")

        self.conv3 = tf.layers.conv2d(inputs=self.conv2_out_2, filters=128, kernel_size=[3, 3], strides=[2, 2],
                                      padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                      name="conv3")

        self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3, training=True, epsilon=1e-5,
                                                             name='batch_norm3')

        self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")

        self.conv4_1 = tf.layers.conv2d(inputs=self.conv3_out, filters=128, kernel_size=[3, 3], strides=[1, 1],
                                        padding="SAME",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name="conv4_1")

        self.conv4_batchnorm_1 = tf.layers.batch_normalization(self.conv4_1, training=True, epsilon=1e-5,
                                                               name='batch_norm4_1')

        self.conv4_out_1 = tf.nn.elu(self.conv4_batchnorm_1, name="conv4_out_1")

        self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_out_1, filters=128, kernel_size=[1, 1], strides=[1, 1],
                                        padding="SAME",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name="conv4_2")

        self.conv4_batchnorm_2 = tf.layers.batch_normalization(self.conv4_2, training=True, epsilon=1e-5,
                                                               name='batch_norm4_2')

        # SE2
        self.se2_1 = tf.keras.layers.GlobalAveragePooling2D()(self.conv4_batchnorm_2)
        self.se2_2 = tf.keras.layers.Dense(self.se2_1.shape[1] // 2)(self.se2_1)
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

        self.flatten = tf.contrib.layers.flatten(self.conv6_out_2)
        cell_size = 256
        s = tf.expand_dims(self.flatten, axis=1,
                           name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
        rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size)
        self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
        outputs, self.final_state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=s, initial_state=self.init_state,
                                                      time_major=True)
        cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation

        self.output_ = tf.layers.dense(inputs=cell_out,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                       units=512,
                                       activation=None,
                                       name="pre_output_internal")
        # critic

        self.tfs_ = self.output_
        lc = tf.layers.dense(self.tfs_, 256, tf.nn.relu, kernel_initializer=w_init, name='lc')
        self.v = tf.layers.dense(lc, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob / (oldpi_prob + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :flattenS_DIM], data[:, flattenS_DIM: flattenS_DIM + 1].ravel(), data[:, -1:]
                s = s.reshape(-1,*S_DIM)
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l_a1 = tf.layers.dense(self.tfs_, 256, tf.nn.relu, trainable=trainable)
            l_a = tf.layers.dense(l_a1, 256, tf.nn.relu, trainable=trainable)
            a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s, cell_state):  # run by a local
        prob_weights, cell_state = self.sess.run([self.pi, self.final_state], feed_dict={self.tfs: s[None, :], self.init_state: cell_state})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, cell_state

    def get_v(self, s,rnn_state_):
        # if s.ndim < 2: s = s[np.newaxis, :]
        s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s, self.init_state: rnn_state_})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),max_num=obj_num)
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            #s = self.env.reset()

            current_num = np.random.randint(obj_num)+1
            self.env.randominit_crowded(current_num)
            s = self.env.getstate_3()
            rnn_state = sess.run(self.ppo.init_state)    # zero rnn state at beginning
            keep_state = rnn_state.copy()       # keep rnn state for updating global net
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []

            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a,rnn_state_ = self.ppo.choose_action(s,rnn_state)

                #s_, r, done, _ = self.env.step(a)
                choice_index = int(a / action_type)
                choice_action = a % action_type
                r, done = self.env.move(choice_index, choice_action)
                s_ = self.env.getstate_3()

                if done: r = -10
                s_stack = s.reshape(-1)
                buffer_s.append(s_stack)
                buffer_a.append(a)
                buffer_r.append(r - 1)  # 0 for not down, -11 for down. Reward engineering
                s = s_
                rnn_state = rnn_state_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    if done:
                        v_s_ = 0  # end of episode
                    else:
                        v_s_ = self.ppo.get_v(s_,rnn_state_)

                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, None]
                    buffer_s, buffer_a, buffer_r = [], [], []

                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

                    if done: break
            if int(GLOBAL_EP / UPDATE_STEP) % 10 == 0 and int(
                    GLOBAL_EP / UPDATE_STEP) > 0:  # !!!!! have been modified!!
                print('model %d saved' % (int(GLOBAL_EP / UPDATE_STEP)))
                saver.save(sess, os.path.join(weight_path, 'model_%d.ckpt' % (int(GLOBAL_EP / UPDATE_STEP))))

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )


if __name__ == '__main__':
    GLOBAL_PPO = PPONet()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode');
    plt.ylabel('Moving reward');
    plt.ion();
    plt.show()
