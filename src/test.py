learning_rate = 0.0002
# Q learning hyperparameters
map_size = 64
# frame_num = 5
obj_num = 25
action_space = 5
state_size = [map_size, map_size, 2]


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

net = DQNetwork17_eval(batch_size=1, seq_len=1, state_size=state_size, learning_rate=learning_rate, num_objects=obj_num,
                       action_space=action_space)

env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(map_size, map_size),
                                                                                     max_num=obj_num)
