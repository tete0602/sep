import sys
sys.path.append("Meta_DDPG_v3")
import tensorflow
import tensorflow as tf
import numpy as np
import gym
import time
import os


#####################  hyper parameters  ####################


###############################  DDPG  ####################################

class DDPG(object):
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.002  # learning rate for critic
    GAMMA = 0.95  # reward discount
    TAU = 0.001  # soft replacement
    MEMORY_CAPACITY = 100000
    BATCH_SIZE = 32
    LAYERS = 4
    learn_step_counter = 0

    # dense是网络层数，units是节点数
    def __init__(self, a_dim, s_dim, a_bound, path, is_train=True, dense=4, units=10, activation='relu'):
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.dense = dense
        self.units = units
        self.is_load_variable = not is_train
        func = {'relu': tf.nn.relu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh}
        self.activation = func[activation]
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        with self.graph.as_default():
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')

            with tf.variable_scope('Actor'):
                self.a = self._build_a(self.S, scope='eval', trainable=True)
                a_ = self._build_a(self.S_, scope='target', trainable=False)
            with tf.variable_scope('Critic'):
                # assign self.a = a in memory when calculating q for td_error,
                # otherwise the self.a is from Actor when updating Actor
                q = self._build_c(self.S, self.a, scope='eval', trainable=True)
                q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

            # networks parameters
            self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
            self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
            self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
            self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

            # target net replacement
            self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                                 for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

            q_target = self.R + self.GAMMA * q_
            # in the feed_dic for the td_error, the self.a should change to actions in memory
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(td_error, var_list=self.ce_params)

            a_loss = - tf.reduce_mean(q)  # maximize the q

            # 记录actor网络的损失
            #        self.grads, _ = tf.gradients(a_loss, self.ae_params)

            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.save_path = 'save/meta/' + path
            if not os.path.exists('/'.join(self.save_path.split('/')[:-1])):
                os.makedirs('/'.join(self.save_path.split('/')[:-1]))
            if self.is_load_variable:
                self.saver.restore(self.sess, self.save_path)
                print('加载参数成功！', self.save_path)

    def assign(self, ae_params, at_params, ce_params, ct_params):
        # self.opt = lambda parme:
        self.sess.run([tf.assign(t, e) for t, e in zip(self.ae_params, ae_params)])
        self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, at_params)])
        self.sess.run([tf.assign(t, e) for t, e in zip(self.ce_params, ce_params)])
        self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, ct_params)])

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.learn_step_counter += 1
        if self.learn_step_counter % 10000 == 0:
            self.saver.save(self.sess, self.save_path)
            print('存储第{}轮参数'.format(self.learn_step_counter))

        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_, done):
        # print(s,'\n',a,'\n',r,'\n',s_)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # s是输入的状态
            net = s
            for i in range(self.dense):
                net = tf.layers.dense(net, self.units, activation=self.activation, name='l_' + str(i),
                                      trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.concat([s, a], axis=1)
            for i in range(self.dense):
                net = tf.layers.dense(net, self.units, activation=self.activation, name='l_' + str(i),
                                      trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def get_apra(self):
        # out = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)for t, e in zip(self.at_params + self.ct_params,
        # self.ae_params + self.ce_params)]

        out1 = self.sess.run(self.ae_params)
        out2 = self.sess.run(self.at_params)
        out3 = self.sess.run(self.ce_params)
        out4 = self.sess.run(self.ct_params)
        return out1,out2,out3,out4

       # return self.sess.run(self.ae_params), self.sess.run(self.at_params), self.sess.run(self.ce_params), self.sess.run(self.ct_params)
    def save_model(self):
        # self.sess.run(tf.global_variables_initializer())
        self.saver.save(self.sess, self.save_path)
        print('==========保存模型成功============result::\n',self.save_path)

    def load_model(self, load_path):
        self.saver.restore(self.sess, load_path)
        print('加载参数成功！')