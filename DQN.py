import sys
sys.path.append("Meta_DDPG_v3")
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DeepQNetwork():
    batch_size = 32
    lr = 0.0001
    memory_size = 100000
    now_index = 0
    learn_step = 1000
    learn_step_counter = 0
    gamma = 0.99

    double_q = True

    def __init__(self, n_action, n_state, path, is_train=True, dense=4, units=10, activation='relu'):
        self.n_state = n_state
        self.is_load_variable = not is_train
        self.e_greedy = 0.1 if is_train else 1.0
        self.e_greedy_finally = 0.95
        self.e_greedy_increase = (self.e_greedy_finally - self.e_greedy) / 30000
        self.n_action = n_action
        self.dense = dense
        self.units = units
        func = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh}
        self.activation = func[activation]
        self.__build_net()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options
        )
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        x = len(var) // 2
        var0, var1 = var[:x], var[-x:]
        self.updata_op = [tf.compat.v1.assign(v1, v0) for v1, v0 in zip(var1, var0)]
        self.s_memory = np.zeros([self.memory_size, self.n_state], dtype=np.float)
        self.ss_memory = np.zeros([self.memory_size, self.n_state], dtype=np.float)
        self.a_memory = np.zeros(self.memory_size, dtype=np.uint8)
        self.r_memory = np.zeros(self.memory_size, dtype=np.float)
        self.dones = np.zeros(self.memory_size, dtype=np.int)
        self.saver = tf.compat.v1.train.Saver()
        self.save_path = 'save/' + path
        if not os.path.exists('/'.join(self.save_path.split('/')[:-1])):
            os.makedirs('/'.join(self.save_path.split('/')[:-1]))
        if self.is_load_variable:
            self.saver.restore(self.sess, self.save_path)
            print('加载参数成功！')

    def __build_net(self):
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_state], name='s')
        self.a = tf.compat.v1.placeholder(tf.float32, [None], name='a')
        self.index = tf.compat.v1.placeholder(tf.int32, [None], name='index')
        with tf.compat.v1.variable_scope('train_variable_0'):
            den = self.s
            for _ in range(self.dense):
                den = tf.layers.Dense(units=self.units, activation=self.activation)(den)
            self.out_0 = tf.layers.Dense(units=self.n_action)(den)
            self.out_00 = tf.nn.softmax(self.out_0, axis=1)

        out_0 = tf.reduce_sum(tf.one_hot(self.index, self.n_action) * self.out_0, 1)
        td_error = out_0 - self.a
        self.loss = tf.reduce_mean(tf.where(tf.abs(td_error) < 1, tf.square(td_error) * 0.5, (tf.abs(td_error) - 0.5)))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.95, epsilon=0.01)

        self.train_op = optimizer.minimize(self.loss)

        self.ss = tf.compat.v1.placeholder(tf.float32, [None, self.n_state], name='ss')
        with tf.compat.v1.variable_scope('train_variable_1'):
            den = self.ss
            for _ in range(self.dense):
                den = tf.layers.Dense(units=self.units, activation=self.activation)(den)
            self.out_1 = tf.layers.dense(inputs=den, units=self.n_action)

    def choose_action(self, s):

        if self.e_greedy > np.random.random():
            s = s[np.newaxis, :]  # 增加batch一维
            out = self.sess.run(self.out_00, feed_dict={self.s: s})
            return out[0]
        else:
            v = np.random.random()
            return np.array([v, 1 - v])

    def store_transition(self, s, a, r, s_,done):

        idx = self.now_index % self.memory_size
        self.s_memory[idx] = s
        self.ss_memory[idx] = s_
        self.a_memory[idx] = a
        self.r_memory[idx] = r
        self.dones[idx] = done
        self.now_index += 1

    def learn(self):
        if self.learn_step_counter % self.learn_step == 0:
            self.sess.run(self.updata_op)
            print('参数更新')
        self.learn_step_counter += 1
        if self.learn_step_counter % 10000 == 0:
            self.saver.save(self.sess, self.save_path)
            print('存储第{}轮参数'.format(self.learn_step_counter))

        self.e_greedy += self.e_greedy_increase if self.e_greedy < self.e_greedy_finally else 0

        up = np.min([self.memory_size, self.now_index])
        memory_index = np.random.randint(0, up, size=self.batch_size)

        batch_s, batch_ss, batch_a, batch_r, dones = self.s_memory[memory_index], self.ss_memory[memory_index], \
                                                     self.a_memory[memory_index], self.r_memory[memory_index], \
                                                     self.dones[memory_index]

        bat = np.arange(self.batch_size)
        q_s, q_ss = self.sess.run([self.out_0, self.out_1], feed_dict={self.s: batch_s, self.ss: batch_ss})
        if self.double_q:
            q_ss_tq1 = self.sess.run(self.out_0, feed_dict={self.s: batch_ss})
            index_tql = np.argmax(q_ss_tq1, 1)
            q_ss_best = q_ss[bat, index_tql]
        else:
            q_ss_best = np.max(q_ss, axis=1)
        q_ss_best *= (1 - dones)

        q_s[bat, batch_a] = batch_r + self.gamma * q_ss_best
        _, loss = self.sess.run([self.train_op, self.loss],
                                feed_dict={self.s: batch_s, self.a: q_s[bat, batch_a], self.index: batch_a})
