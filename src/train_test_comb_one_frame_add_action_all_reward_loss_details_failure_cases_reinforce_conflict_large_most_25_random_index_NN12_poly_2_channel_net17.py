import tensorflow as tf
import numpy as np
import os
from env import ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly
from copy import deepcopy
import time
from lstm import DQNetwork17, DQNetwork17_eval
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
    net_train = DQNetwork17(batch_size=batch_size, seq_len=seq_len, state_size=state_size, learning_rate=learning_rate,
                            num_objects=obj_num, action_space=action_space)
    net = DQNetwork17_eval(batch_size=1, seq_len=1, state_size=state_size, learning_rate=learning_rate,
                           num_objects=obj_num, action_space=action_space)
    net_infer = DQNetwork17_eval(batch_size=batch_size, seq_len=seq_len, state_size=state_size,
                                 learning_rate=learning_rate, num_objects=obj_num, action_space=action_space)

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
    for i in range(action_space):
        tf.summary.scalar("Loss of Action %d:" % (i + 1), net_train.loss_details[i])

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

        for a in range(action_size):
            env_copy = deepcopy(env)
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

            Qs, h_state = sess.run([net.output, net.state_out],
                                   feed_dict={net.inputs_: state_.reshape((1, 1, *state_.shape)),
                                              net.finish_tag: finish_tag.reshape((1, 1, *finish_tag.shape)),
                                              net.state_in: h_state})
            Qs = Qs.squeeze()
            # Take the biggest Q value (= the best action)

            env_copy = deepcopy(env)
            while True:
                choice = np.argmax(Qs)
                choice_index = int(choice / action_space)
                choice_action = choice % action_space
                reward, done = env_copy.move(choice_index, choice_action)

                if done == -1:
                    Qs[choice] = -1000000000
                    continue

                break

            action = np.zeros(action_space * obj_num)
            action[choice] = 1

        action_index = np.argmax(action)
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
                # print(next_states_mb[:, :, a].shape)
                # print(finish_tag_next_mb[:, :, a].shape)
                # print(Qs_next_state.shape)
                # quit()
                Qs = np.max(Qs_next_state, axis=-1)
                target_Qs_batch.append(Qs)

            # [bs, time, a]
            targets_mb = np.array(target_Qs_batch).transpose([1, 2, 0])

            targets_mb = rewards + gamma * dones_mb * targets_mb

            # for i in range(batch_size):
            #     done = dones_mb[i]
            #     if done == 1:
            #         # target_Qs_batch.append(rewards_mb[i])
            #         target_Qs.append(rewards_mb[i,a])
            #     else:
            #         target = rewards_mb[i,a] + gamma * np.max(Qs_next_state[i])
            #         # target_Qs_batch.append(target)
            #         target_Qs.append(target)

            # target_Qs_batch.append(target_Qs)

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


def test_mover_64_net(env, net, sess):  # whole model
    from visualization import convert
    from visualization import convert_to_img
    from PIL import Image
    from MCTS import MCT
    # import imageio

    gamma = 0.95

    '''
        HyperParameters
    '''

    obj_num = 25
    action_space = 5
    action_size = obj_num * action_space
    dim_h = 256
    batch_size = 1

    z_state = np.zeros([batch_size, dim_h])
    # init_state = (z_state, z_state)
    init_state = (z_state, z_state, z_state, z_state)

    frames = []
    last_list = []

    time_dict = {}

    def start(id_=0):
        time_dict[id_] = time.time()

    def end(id_=0):
        return time.time() - time_dict[id_]

    finished_test = False
    total_reward = 0
    s = 0
    input_map = deepcopy(env.getmap())
    target_map = env.gettargetmap()

    tree = MCT()
    tree.root.state = deepcopy(env)
    t = time.time()
    tree.setactionsize(action_size)
    cflag = False

    # h_state = sess.run(net.state_out, feed_dict = {net.inputs_: state.reshape(1,1,*state.shape), net.finish_tag: tag.reshape(1,1,*tag.shape), net.state_in:init_state})

    tree.root.h_state = init_state  # which represents the last hidden state

    length = 0

    frame_cnt = 0
    while (not (finished_test == 1)) and s < 80:
        s += 1
        max_depth = 0
        h_state = tree.root.h_state
        # env = tree.root.state

        state = env.getstate_3()
        tag = env.getfinished()

        h_state = sess.run(net.state_out, feed_dict={net.inputs_: state.reshape(1, 1, *state.shape),
                                                     net.finish_tag: tag.reshape(1, 1, *tag.shape),
                                                     net.state_in: h_state})

        for i in range(action_size):
            # print('action_now',i)
            if not tree.haschild(tree.root.id, i):
                item = int(i / action_space)
                direction = i % action_space
                env_copy = deepcopy(env)
                reward, done = env_copy.move(item, direction)
                succ, child_id = tree.expansion(tree.root.id, i, reward, env_copy, done)

                # print('item',item,'direction',direction,'succ',succ)
                if succ:

                    # simulation
                    if done != 1:
                        policy = 0
                        cnt = 0
                        reward_sum_a = 0
                        env_sim = deepcopy(env_copy)
                        end_flag = False
                        h_state_t = h_state

                        while (not (end_flag == 1)) and cnt < 20:
                            cnt += 1

                            state = env_sim.getstate_3()
                            tag = env_sim.getfinished()
                            # finishtag = np.pad(finish_tag,(0,obj_num-exp),"constant")
                            # print(finishtag.shape,obj_num,exp,finish_tag.shape)

                            Qs, h_state_t = sess.run([net.output, net.state_out],
                                                     feed_dict={net.inputs_: state.reshape(1, 1, *state.shape),
                                                                net.finish_tag: tag.reshape(1, 1, *tag.shape),
                                                                net.state_in: h_state_t})

                            Qs = Qs.squeeze()

                            while True:
                                action = np.argmax(Qs)
                                item = int(action / action_space)
                                direction = action % action_space
                                reward, done = env_sim.move(item, direction)

                                if done != -1:
                                    break

                                Qs[action] = -np.inf

                            reward_sum_a += reward * pow(gamma, cnt - 1)
                            end_flag = done

                        reward_sum = reward_sum_a
                        policy = 0

                        tree.nodedict[child_id].value = reward_sum
                        tree.nodedict[child_id].policy = policy
                        tree.nodedict[child_id].h_state = h_state

                        if reward_sum > 60:
                            print('wa done!', reward_sum, 'policy', policy)
                            cflag = True

                    tree.backpropagation(child_id)

        if cflag:
            C = 0.1
        else:
            C = 0.2

        cflag = False
        print(C)

        node_cnt = 0
        t_io = 0
        t_nn = 0
        t_tree = 0
        t_copy = 0
        start('total')
        while node_cnt < 200:
            start('tree')
            _id = tree.selection(C)
            t_tree += end('tree')
            policy = tree.nodedict[_id].policy
            start('copy')
            env_copy = deepcopy(tree.getstate(_id))
            t_copy += end('copy')
            h_state = tree.nodedict[_id].h_state

            if policy == 0:
                start('io')
                state = env_copy.getstate_3()
                tag = env_copy.getfinished()
                t_io += end('io')

                start('nn')
                Qs, h_state = sess.run([net.output, net.state_out],
                                       feed_dict={net.inputs_: state.reshape(1, 1, *state.shape),
                                                  net.finish_tag: tag.reshape(1, 1, *tag.shape), net.state_in: h_state})
                t_nn += end('nn')

                Qs = Qs.squeeze()

                # while True:
                #     action = np.argmax(Qs)
                #     if tree.haschild(_id, action):
                #         Qs[action] = -np.inf
                #         continue

                #     break

            start('tree')
            empty_actions = tree.getemptyactions(_id)
            t_tree += end('tree')
            while True:

                action = np.argmax(Qs)
                if not action in empty_actions:
                    Qs[action] = -np.inf
                    continue

                start('io')
                item = int(action / action_space)
                direction = action % action_space
                reward, done = env_copy.move(item, direction)
                t_io += end('io')

                start('tree')
                succ, child_id = tree.expansion(_id, action, reward, env_copy, done)
                t_tree += end('tree')
                break

            # simulation
            if succ:
                node_cnt += 1
                if done != 1:

                    # value = np.max(Qs)

                    cnt = 0
                    reward_sum_a = 0
                    start('copy')
                    env_sim = deepcopy(env_copy)
                    t_copy += end('copy')
                    end_flag = False
                    h_state_t = h_state

                    while (not (end_flag == 1)) and cnt < 20:
                        cnt += 1

                        start('io')
                        state = env_sim.getstate_3()
                        tag = env_sim.getfinished()
                        t_io += end('io')

                        start('nn')
                        Qs, h_state_t = sess.run([net.output, net.state_out],
                                                 feed_dict={net.inputs_: state.reshape(1, 1, *state.shape),
                                                            net.finish_tag: tag.reshape(1, 1, *tag.shape),
                                                            net.state_in: h_state_t})
                        t_nn += end('nn')

                        Qs = Qs.squeeze()

                        while True:
                            action = np.argmax(Qs)
                            item = int(action / action_space)
                            direction = action % action_space
                            start('io')
                            reward, done = env_sim.move(item, direction)
                            t_io += end('io')

                            if done != -1:
                                break

                            Qs[action] = -np.inf

                        reward_sum_a += reward * pow(gamma, cnt - 1)
                        end_flag = done

                    reward_sum = reward_sum_a
                    policy = 0

                    tree.nodedict[child_id].value = reward_sum
                    tree.nodedict[child_id].policy = policy
                    tree.nodedict[child_id].h_state = h_state

                    if reward_sum > 60:
                        print('wa done!', reward_sum, 'policy', policy)
                        cflag = True

                max_depth = max([max_depth, tree.nodedict[child_id].depth])
                start('tree')
                tree.backpropagation(child_id)
                t_tree += end('tree')

        t_total = end('total')
        action = tree.root.best
        item = int(action / action_space)
        direction = action % action_space
        prestate = env.cstate[item]
        prepos = env.pos[item]
        reward, done = env.move(item, direction)
        finished_test = done
        if direction == 4:
            length += route_length(env.getlastroute())
        else:
            length += env.last_steps

        print('step', s, 'times', tree.nodedict[tree.root.childs[action]].times, 'id', tree.root.id, 'max_depth',
              max_depth - tree.root.depth, 'value', tree.root.value, 'best depth', tree.getbestdepth(), 'io time', t_io,
              'nn time', t_nn, 'copy time', t_copy, 'tree time', t_tree, 'total time', t_total, 'policy',
              tree.root.policy, 'current reward', total_reward)
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
            print('id', node.id, 'value', node.value, 'reward', node.reward, 'best', best, 'depth',
                  node.depth - tree.root.depth)
            last_list.append(node.id)

        # tree.print_structure()

        tree.nextstep(action)
        total_reward += reward

        # move_sequence = []
        # if direction == 4:
        #     route = env.getlastroute()
        #     route_map = np.zeros_like(input_map)
        #     for node in route:
        #         x,y,st,d = node
        #         route_map[int(x),int(y)] = 1

        #     for node in route:
        #         x,y,st,ds = node
        #         imap = env.getcleanmap(item)
        #         ps,bx = env.getitem(item, st)
        #         # x = max(0, x - 0.5*(bx[1,0]-bx[0,0]+1))
        #         # y = max(0, y - 0.5*(bx[1,1]-bx[0,1]+1))

        #         for p in ps:
        #             xx,yy = p
        #             xx += x
        #             yy += y
        #             xx = int(xx)
        #             yy = int(yy)
        #             if xx >= 0 and xx < imap.shape[0] and yy >= 0 and yy < imap.shape[1]:
        #                 imap[xx,yy] = item + 2

        #         img = convert_to_img(imap, target_map, route_map)
        #         im = Image.fromarray(img)
        #         im.save('%s/case%d_%d.png'%(save_path,idx,frame_cnt))
        #         frame_cnt += 1

        #     # img = convert_to_img(input_map,target_map,route_map)
        #     # frames.append(img)

        # # input_map = deepcopy(env.getmap())
        # # img = convert_to_img(input_map,target_map,np.zeros_like(input_map))
        # # frames.append(img)
        # input_map = deepcopy(env.getmap())
        # img = convert_to_img(input_map, target_map, np.zeros_like(input_map))
        # im = Image.fromarray(img)
        # im.save('%s/case%d_%d.png'%(save_path,idx,frame_cnt))
        # frame_cnt += 1

        # if direction == 4:
        #     route = env.getlastroute()

        #     # xxx,yyy = prepos
        #     for node in route:
        #         x,y,st,ds = node
        #         x /= 64
        #         y /= 64
        #         if st != prestate:
        #             if ds == 0:
        #                 angle = (st-prestate+env.bin)%env.bin
        #             else:
        #                 angle = (prestate-st+env.bin)%env.bin

        #             move_sequence.append('%d,r,%d\n'%(item,angle))

        #         move_sequence.append('%d,t,%.3f,%.3f\n'%(item,y,x))
        #         prestate = st

        # else:
        #     xxx, yyy = env.pos[item]
        #     xxx /= 64
        #     yyy /= 64
        #     move_sequence.append('%d,t,%.3f,%.3f\n'%(item,yyy,xxx))

        # with open('./%s/livingroom_sequence.txt'%save_path,'a') as fp:
        #     for a_move in move_sequence:
        #         fp.write(a_move)

    if finished_test == 1:
        print('Finished! Reward:', total_reward, 'Steps:', s, 'TL', length)
    else:
        print('Failed Reward:', total_reward, 'Steps:', s, 'TL', length)



if __name__ == '__main__':

    # train
    # train_comb_one_frame_add_action_all_reward_loss_details_failure_cases_reinforce_conflict_large_most_25_random_index_NN12_poly_2_channel_net17()


    #test
    # Q learning hyperparameters
    learning_rate = 0.0002
    map_size = 64
    # frame_num = 5
    obj_num = 25
    action_space = 5
    state_size = [map_size, map_size, 2]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    net = DQNetwork17_eval(batch_size=1, seq_len=1, state_size=state_size, learning_rate=learning_rate,
                           num_objects=obj_num,
                           action_space=action_space)

    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),
                                                                                         max_num=obj_num)

    env.randominit_crowded(np.random.randint(obj_num) + 1)

    test_mover_64_net(env, net, sess)
