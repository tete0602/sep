# encoding=utf-8
# ------调用DDPG模型-------
import sys

from DDPG import DDPG
# ------调用动力系统例子-------
from Env import get_Env, get_singleTest

import joblib
from sklearn.preprocessing import PolynomialFeatures
import random
import numpy as np
from matplotlib import pyplot as plt
# from fit_poly import fit_poly
import time
from collections import deque
# from MetaNet import MetaNet
from copy import deepcopy
import os

from Env2 import ex_9_dim
# from Env2 import ex_12_dim
# from Env2 import ex_6_dim
# from Env2 import ex_4_dim
# from Env2 import ex_2_dim

ddpg = True

# tasks_num = 2

# tasks_num_test = 2
train_num = 5
test_num = 5

epochs = 50
trajectory_num = 50
step_num = 300
# epochs=10



def train():


    # global safe_time
    # --------------------------------------------------------------------------------------------------

    # 随机从数据集中选取一个system
    env_meta = ex_9_dim(random.randint(0, 49))
    # 根据选取的system定义一个DDPG类型的meta网络模型
    model_meta = DDPG(1, a_bound=env_meta.u, s_dim=env_meta.n_obs, is_train=True, path=env_meta.path,
                      units=env_meta.units, dense=env_meta.dense, activation=env_meta.activation)
    # --------------------------------------------------------------------------------------------------
    tot = 0
    # 安全轨迹数
    safe_time = 0
    # 每条轨迹的长度
    mxlen = 50
    # deque()双向队列----类似于list的容器，可以快速的在队列头部和尾部添加、删除元素
    dq = deque(maxlen=mxlen)
    # var 表示随机正态分布的标准差，越大分布越均匀
    var = 3
    # env = get_Env(0)
    # agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path, units=env.units, graph='exp1',
    # dense=env.dense, activation=env.activation)
    models = []
    env_train=[]
    # --------------------------------------------------------------------------------------------------
    outputText = './outputText/50_mate_message_9.txt'

    # meta train
    #总的训练时长
    start_tot = time.time()
    for t in range(epochs):
        # print('sep')
        # print(epochs)
        print('===========新的epoch===========', t)
        print('===========使用训练资料===========')
        for i in range(train_num):
            env = ex_9_dim(random.randint(0, 49))
            agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
                         units=env.units, dense=env.dense, activation=env.activation)
            env_train.append(env)
            models.append(agent)
            # print(env_train)
            # print(models)
        # 对于每一个训练任务
        for i in range(train_num):
            # 将初始元网络模型参数赋值给第一个训练任务
            # models[i].assign(model_meta.get_apra())
            out1, out2, out3, out4 = model_meta.get_apra()
            models[i].assign(out1, out2, out3, out4)
            # models[i] = deepcopy(model_meta)
            # 随机选择一个训练任务
            env = env_train[i]
            # print('输出范围：', action_value)
            print('==========train：第', i, '个example===========')
            # 对于每一条轨迹
            for k in range(trajectory_num):
                # reset()重置函数-----重新选择初始点
                s = env.reset()
                # print('初始点:', s)
                reward = 0
                # count = 0
                # 对于每条轨迹中的每一步
                # safe_time=0
                for p in range(step_num):
                    dq.append(sum(env.s ** 2))
                    a = models[i].choose_action(s)
                    # a表示正态分布的均值，以a为均值，var为标准差，随机取值
                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]

                    # add randomness to action selection for exploration
                    # 为探索的动作选择添加随机性
                    s_, r, done, info = env.step(a_v)
                    if info[1]:
                        print('李导数不满足!!')
                    if not ddpg:
                        # np.argmax()沿给定轴返回最大元素索引
                        a_v = np.argmax(a)
                    models[i].store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('不安全')
                        safe_time = 0
                        # ‘//’整数除法，返回商的整数部分
                        tot = tot // 2000 * 2000 + 1010
                    if tot % 2000 == 0:
                        done = 1
                    tot += 1
                    if tot > 1000:
                        if tot % 10 == 0:
                            var *= .9995
                        # 学习
                        models[i].learn()
                    # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                    if done or (len(dq) == mxlen and np.var(dq) < 1e-10):
                        print('reward:', reward, 'info:', info,
                              (' Explore:', var) if ddpg else ('e_greedy:', models[i].e_greedy))
                        break
                    s = s_.copy()

            trajectory_num_meta = 50
            print('===========开始元更新===========epoch：', t)
            print('===========使用测试资料===========')
            testTask = random.randint(0, 49)
            env_test= ex_9_dim(testTask)
            # s = env.reset()
            # apra = models[i].get_apra()
            # meta_grads = []
            print('===========使用模型' + str(i) + '测试任务' + str(testTask) + '============')
            reward = 0
            agent_test = DDPG(1, a_bound=env_test.u, s_dim=env_test.n_obs, is_train=True, path=env_test.path,
                         units=env_test.units, dense=env_test.dense, activation=env_test.activation)

            out1, out2, out3, out4 = models[i].get_apra()
            agent_test.assign(out1, out2, out3, out4)

            start = time.time()
            for k in range(trajectory_num_meta):
                s = env.reset()
                # print('初始点:', s)
                # count = 0
                for p in range(step_num):
                    a = agent_test.choose_action(s)
                    # grad = models[i].get_grad()
                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                    s_, r, done, info = env.step(a_v)
                    model_meta.store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('不安全')
                        safe_time = 0
                        tot = tot // 2000 * 2000 + 1010
                    if tot % 2000 == 0:
                        done = 1
                    tot += 1
                    if tot > 1000:
                        if tot % 10 == 0:
                            var *= .9995
                        model_meta.learn()
                    # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                    if done or (len(dq) == mxlen and np.var(dq) < 1e-10):
                        print('reward:', reward, 'info:', info,
                              (' Explore:', var) if ddpg else ('e_greedy:', 1))
                        break
                    if info[1] == True:
                        print('李导数不满足！')
                    s = s_.copy()
                safe_time+= 1
            print("meta: reward::", reward / trajectory_num_meta)
            print('安全的轨迹数：', safe_time)
            # 每个example 设定了上限，2维的一般只采样50条轨迹
            end = time.time()
            if (safe_time >= 50) or (end - start) / 60 /60 > 1:
                ti = (end - start)
                ti = round(ti, 2)
                print('testTask{},训练用时：{}s'.format(testTask, ti))
                # outputText = './outputText/mate_message_4.txt'
                # if not os.path.exists('./outputText'):
                #     os.makedirs('./outputText')

                # with open(outputText, 'a', encoding='utf-8') as f:
                    # f.write('训练集中第'+ str(i) + '个任务')
                    # f.write('层数:{}\n每层{}个结点\n{}激活函数\n'.format(env.dense, env.units, env.activation))
                    # f.write('训练耗时：' + str(ti) + 's' + '\n')
                    # f.write('训练轮数:' + str(t + 1) + '\n')
                    # f.write('第{}轮'.format(index) + '\n')
                    # f.write('安全的轨迹数：' + str(safe_time) + '\n')
                break

    model_meta.save_model()
    end_tot = time.time()
    print('总训练用时：{}s'.format(end_tot - start_tot))
    with open(outputText, 'a', encoding='utf-8') as f:
        # f.write('第{}个env'.format(index)+'\n')
        f.write('总训练用时：{}s'.format(end_tot - start_tot) + '\n')

    return model_meta


# def train_with_certain_times():
#     os.remove('./outputText/mate_message_12.txt')
#     for i in range(50):
#         train(i)
#         test(i)

def train_with_envlist_maml(envlist,tot):
    # filename = './outputText/50_mate_message_9.txt'
    filename = 'parameter_test/dim_2.txt'
    if os.path.exists(filename):
        os.remove(filename)

    model_meta = train()
    for i, item in enumerate(envlist):
        test(i, item, model_meta)
        env.s = np.array([np.random.random() - 0.5 for _ in range(2)])

    # lock.release();


def test(index, env_test, model_meta):
    # model = joblib.load('model/ex{}{}.model'.format(env.id, '_with_lidao' if env.is_lidao else ''))
    # env_meta = get_singleTest(1)
    env_mate = ex_9_dim(index)
    # model_meta = DDPG(1, a_bound=env_meta.u, s_dim=env_meta.n_obs, is_train=False, path=env_meta.path,
    #                   units=env_meta.units, dense=env_meta.dense, activation=env_meta.activation)
    model_meta=model_meta
    task_test_num = 1
    for i in range(task_test_num):
        print('===========开始元测试===========')
        print('===========使用测试集进行训练===========', i)
        print('===========使用metaNet训练一个新的example===========', i)
        # env =  get_singleTest(1)
        # env=ex_6_dim(random.randint(0,49))
        # print(env_test)
        env = env_test

        agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
                     units=env.units, dense=env.dense, activation=env.activation)

        tot = 0
        start = time.time()
        safe_time = 0
        mxlen = 50
        dq = deque(maxlen=mxlen)
        var = 3
        action_value = np.linspace(-env.u, env.u, 2)
        print('输出范围：', action_value)
        out1, out2, out3, out4 = model_meta.get_apra()
        agent.assign(out1, out2, out3, out4)
        for t in range(10000):
            s = env.reset()
            print('初始点:', s)
            reward = 0
            count = 0
            dq = deque(maxlen=mxlen)
            while True:
                count += 1
                dq.append(sum(env.s ** 2))

                a = agent.choose_action(s)
                a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                # print('chose action a:', a)
                # print('chose action a[0:', a[0])
                # print('chose action a[1:', a[1])
                # add randomness to action selection for exploration
                # print(np.clip(np.random.normal(a[0], var), -env.u, env.u))
                '''多维u
                a_v = []
                u1 = np.clip(np.random.normal(a[0], var), -env.u, env.u)
                u2 = np.clip(np.random.normal(a[1], var), -env.u, env.u)
                a_v.append(u1)
                a_v.append(u2)
                '''

                # print('chose action a_v:', a_v)
                # print('v:',a_v)
                s_, r, is_unsafe, info = env.step(a_v)
                # print('done:', done)
                if info[1]:
                    print('李导数不满足!!')
                if not ddpg:
                    a_v = np.argmax(a)
                agent.store_transition(s, a_v, r, s_, is_unsafe)
                reward += r
                if is_unsafe:
                    print('======不安全=====')
                    print('不安全的点:', s_)
                    print('reward:', r)
                    safe_time = 0
                    tot = tot // 2000 * 2000 + 1010
                    # outputText = './outputText/mate_message_4.txt'
                    # with open(outputText, 'a', encoding='utf-8') as f:
                    #     f.write('测试任务不安全！！:'+'\n')
                    #     f.write('不安全的点:' + str(s_) + '\n')
                    break
                if tot % 2000 == 0:
                    is_unsafe = 1
                tot += 1
                if tot > 1000:
                    if tot % 10 == 0:
                        var *= .9995
                    # 学习
                    agent.learn()
                # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                if is_unsafe or (len(dq) == mxlen and np.var(dq) < 1e-10):
                    print('reward:', reward, 'info:', info,
                          (' Explore:', var) if ddpg else ('e_greedy:', agent.e_greedy))

                    break
                s = s_.copy()
            safe_time += 1
            print('安全的轨迹数：', safe_time)
            # k 表示连续k条轨迹安全则训练结束
            end = time.time()
            if (safe_time >= 50):
                # end = time.time()
                ti = (end - start)
                ti = round(ti, 2)
                print('测试集{}的训练用时：{}s'.format(i, ti))
                print('测试集{}的训练轮数：{}'.format(i, t))
                print('=====连续{}条轨迹安全，测试通过=======', format(50))
                agent.save_model()
                # draw(env,i,agent)
                # draw(env)
                # testest(agent,env,i)
                print('层数:{}\n每层{}个结点\n{}激活函数\n'.format(env.dense, env.units, env.activation))
                outputText = './outputText/50_mate_message_9.txt'
                with open(outputText, 'a', encoding='utf-8') as f:
                    f.write('第{}个env'.format(index) + '\n')
                    f.write('安全轨迹数：' + str(safe_time) + '\n')
                    f.write('参数测试耗时：' + str(ti) + 's' + '\n')
                    f.write('不安全域:' + str(env.U_zones) + '\n')
                    f.write('安全区域:' + str(env.I_zones) + '\n')
                    f.write('-----------------------------------\n')
                print('训练耗时：' + str(ti) + 's' + '\n')
                print('训练轮数:' + str(t + 1) + '\n')

                break
            else:
                if ((end - start) / 60/60 > 1):
                    outputText = './outputText/50_mate_message_9.txt'
                    with open(outputText, 'a', encoding='utf-8') as f:
                        f.write('第{}轮'.format(index) + '-----测试失败----' + '\n')
                        f.write('-----------------------------------\n')
                    break


def testest(agent, env, i):
    print('=============test test===========')
    tot = 0
    # start = time.time()
    safe_time = 0
    mxlen = 50
    # dq = deque(maxlen=mxlen)
    var = 3
    verifier_time_start = time.time()
    for t in range(10000):
        s = env.reset()
        # print('初始点:', s)
        reward = 0
        count = 0
        dq = deque(maxlen=mxlen)
        step = 0
        step_end = 0
        while True:

            count += 1
            dq.append(sum(env.s ** 2))

            a = agent.choose_action(s)
            a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
            # print('chose action a:', a)
            # print('chose action a[0:', a[0])
            # print('chose action a[1:', a[1])
            # add randomness to action selection for exploration
            # print(np.clip(np.random.normal(a[0], var), -env.u, env.u))
            '''多维u
            a_v = []
            u1 = np.clip(np.random.normal(a[0], var), -env.u, env.u)
            u2 = np.clip(np.random.normal(a[1], var), -env.u, env.u)
            a_v.append(u1)
            a_v.append(u2)
            '''

            # print('chose action a_v:', a_v)
            # print('v:',a_v)
            s_, r, is_unsafe, info = env.step(a_v)
            # print('done:', done)
            if info[1]:
                print('李导数不满足!!')
            reward += r
            if is_unsafe:
                print('不安全点', s_)
                print('=============不安全，验证不通过===========')
                safe_time = 0
                return
            #  tot = tot // 2000 * 2000 + 1010
            if step >= 200:
                step_end = 1
            step += 1

            # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
            if is_unsafe or step_end or (len(dq) == mxlen and np.var(dq) < 1e-10):
                print('第{}条轨迹,轨迹长度{}'.format(t, step))
                print('reward:', reward, 'info:', info,
                      (' Explore:', var) if ddpg else ('e_greedy:', agent.e_greedy))

                break
            s = s_.copy()
        safe_time += 1
        print('安全的轨迹数：', safe_time)
        end = time.time()
        # k 表示连续k条轨迹安全则测试结束
        if (safe_time >= 20):
            # ti = (end - start)
            # ti = round(ti, 2)
            verifier_time_end = time.time()
            ti = verifier_time_end - verifier_time_start
            ti = round(ti, 2)
            print('=======测试通过========{}'.format(i))
            print('验证用时：{}s'.format(ti))
            agent.save_model()
            # draw(env, i, agent)
            print('层数:{}\n每层{}个结点\n{}激活函数\n'.format(env.dense, env.units, env.activation))
            print('训练耗时：' + str(ti) + 's' + '\n')
            print('训练轮数:' + str(t + 1) + '\n')
            # draw(env)
            break


def draw(env):
    print('=========开始画图==========')
    X1 = []
    X2 = []
    # model = joblib.load('model/ex{}{}.model'.format(env.id, '_with_lidao' if env.is_lidao else ''))
    agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=False, path=env.path,
                 units=env.units, dense=env.dense, activation=env.activation)
    point_num = 10
    if env.I_zones.shape == 'box':
        up = env.I_zones.up[0]
        down = env.I_zones.low[0]
        left = env.I_zones.low[1]
        right = env.I_zones.up[1]
        x = np.linspace(left, right, point_num)
        y = np.linspace(down, up, point_num)
    else:
        r = np.sqrt(env.I_zones.r)
        x = np.linspace(env.I_zones.center[0] - r, env.I_zones.center[0] + r, point_num)
        y = np.linspace(env.I_zones.center[1] - r, env.I_zones.center[1] + r, point_num)
    # 随机采样一个点
    for i in range(len(x)):
        for j in range(len(y)):
            s = np.array([x[i], y[j]])
            if env.I_zones.shape == 'ball' and (sum((s - env.I_zones.center) ** 2) > env.I_zones.r):
                continue

            # print(sum((s - env.I_zones.center) ** 2) > env.I_zones.r)
            env.unisample(s)
            tot = 0
            # print('初始状态:', s)
            mxlen = 50
            dq = deque(maxlen=mxlen)
            X1.clear()
            X2.clear()
            tot = 0
            while True:
                tot += 1
                # a = agent.choose_action(s)
                # a_v = np.dot(a, action_value.T)
                X1.append(s[0])
                X2.append(s[1])

                a = agent.choose_action(s)

                s_, r, done, info = env.step(a[0])

                if info[1] == True:
                    print('李导数不满足！')
                if info[0] == True:
                    print('进入非安全区域！')
                s = s_.copy()
                dq.append(sum(env.s ** 2))
                if len(dq) == mxlen and np.var(dq) < 1e-10:
                    break
                if tot > 4000:
                    break
                if done:
                    print('=========进入非安全区域!=========')
                    print(s_)
            # print('轨迹轨迹长度:', len(X1))
            plt.plot(X1, X2)

    if env.D_zones.shape == 'box':
        plt.xlim(env.D_zones.low[0], env.D_zones.up[0])
        plt.ylim(env.D_zones.low[1], env.D_zones.up[1])
    else:
        r = np.sqrt(env.D_zones.r)
        plt.xlim(env.D_zones.center[0] - r, env.D_zones.center[0] + r)
        plt.ylim(env.D_zones.center[1] - r, env.D_zones.center[1] + r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.D_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.D_zones.center[1] for v in thta]
        plt.plot(x, y)

    if env.I_zones.shape == 'ball':
        r = np.sqrt(env.I_zones.r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.I_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.I_zones.center[1] for v in thta]
        plt.plot(x, y)
    else:
        up = env.I_zones.up[0]
        down = env.I_zones.low[0]
        left = env.I_zones.low[1]
        right = env.I_zones.up[1]
        x = [up, up, down, down, up]
        y = [right, left, left, right, right]
        plt.plot(x, y)
    if env.U_zones.shape == 'ball':
        r = np.sqrt(env.U_zones.r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.U_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.U_zones.center[1] for v in thta]
        plt.plot(x, y)
    else:
        up = env.U_zones.up[0]
        down = env.U_zones.low[0]
        left = env.U_zones.low[1]
        right = env.U_zones.up[1]
        x = [up, up, down, down, up]
        y = [right, left, left, right, right]
        plt.plot(x, y)
    plt.grid(True)
    plt.show()
    savepath = 'ddpg{}.png'.format(4)
    plt.savefig(savepath)


def draw_line():
    # model = joblib.load('model/ex{}{}.model'.format(env.id, '_with_lidao' if env.is_lidao else ''))
    env_meta = get_testEnv_9dim(0)

    model_meta = DDPG(1, a_bound=env_meta.u, s_dim=env_meta.n_obs, is_train=False, path='dim2/metamodel',
                      units=env_meta.units, dense=env_meta.dense, activation=env_meta.activation)
    tasks_num_test = 1
    output0 = deque(maxlen=10005)
    output1 = deque(maxlen=10005)
    for i in range(tasks_num_test):
        print('===========开始元测试===========')
        print('===========使用测试集进行训练===========', i)
        print('===========使用metaNet训练一个新的example===========', i)
        env = get_singleTest(4)
        agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
                     units=env.units, dense=env.dense, activation=env.activation)
        tot = 0
        start = time.time()
        safe_time = 0
        mxlen = 50
        dq = deque(maxlen=mxlen)
        var = 3
        action_value = np.linspace(-env.u, env.u, 2)
        print('输出范围：', action_value)
        out1, out2, out3, out4 = model_meta.get_apra()
        agent.assign(out1, out2, out3, out4)
        for t in range(100):
            s = env.reset()
            print('初始点:', s)
            reward = 0
            count = 0
            dq = deque(maxlen=mxlen)
            while True:
                count += 1
                dq.append(sum(env.s ** 2))

                a = agent.choose_action(s)
                a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                # print('chose action a:', a)
                # print('chose action a[0:', a[0])
                # print('chose action a[1:', a[1])
                # add randomness to action selection for exploration
                # print(np.clip(np.random.normal(a[0], var), -env.u, env.u))
                '''多维u
                a_v = []
                u1 = np.clip(np.random.normal(a[0], var), -env.u, env.u)
                u2 = np.clip(np.random.normal(a[1], var), -env.u, env.u)
                a_v.append(u1)
                a_v.append(u2)
                '''

                # print('chose action a_v:', a_v)
                # print('v:',a_v)
                s_, r, is_unsafe, info = env.step(a_v)
                # print('done:', done)
                if info[1]:
                    print('李导数不满足!!')
                if not ddpg:
                    a_v = np.argmax(a)
                agent.store_transition(s, a_v, r, s_, is_unsafe)
                reward += r
                if is_unsafe:
                    print('======不安全=====')
                    print('不安全的点:', s_)
                    safe_time = 0
                    tot = tot // 2000 * 2000 + 1010
                    break
                if tot % 2000 == 0:
                    is_unsafe = 1
                tot += 1
                if tot > 1000:
                    if tot % 10 == 0:
                        var *= .9995
                    # 学习
                    agent.learn()
                # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                if is_unsafe or (len(dq) == mxlen and np.var(dq) < 1e-10):
                    print('reward:', reward, 'info:', info,
                          (' Explore:', var) if ddpg else ('e_greedy:', agent.e_greedy))

                    break
                s = s_.copy()
            safe_time += 1
            print('安全的轨迹数：', safe_time)
            # k 表示连续k条轨迹安全则训练结束
            end = time.time()
            # ===========随机采样n个点求成功率================
            safe = 0
            unsafe = 0
            safe = 0
            point_num = 10
            if env.I_zones.shape == 'box':
                x = np.linspace(env.I_zones.low[0], env.I_zones.up[0], point_num)
                y = np.linspace(env.I_zones.low[1], env.I_zones.up[1], point_num)
            else:
                r = np.sqrt(env.I_zones.r)
                x = np.linspace(env.I_zones.center[0] - r, env.I_zones.center[0] + r, point_num)
                y = np.linspace(env.I_zones.center[1] - r, env.I_zones.center[1] + r, point_num)
            # 随机采样一个点
            tot_num = 0
            for i in range(len(x)):
                for j in range(len(y)):
                    xi = x[i] + (np.random.rand(1)[0] - 0.5)
                    yj = y[i] + (np.random.rand(1)[0] - 0.5)
                    s = np.array([xi, yj])
                    if env.I_zones.shape == 'ball' and (sum((s - env.I_zones.center) ** 2) > env.I_zones.r):
                        continue

                    # print(sum((s - env.I_zones.center) ** 2) > env.I_zones.r)
                    env.unisample(s)
                    tot = 0
                    # print('初始状态:', s)
                    mxlen = 100
                    dq = deque(maxlen=mxlen)
                    tot = 0  # 轨迹长度
                    tot_num += 1
                    while True:
                        tot += 1
                        # a = agent.choose_action(s)
                        # a_v = np.dot(a, action_value.T)
                        a = agent.choose_action(s)
                        s_, r, done, info = env.step(a[0])
                        # if info[1] == True:
                        #     print('李导数不满足！')
                        # if info[0] == True:
                        #     print('进入非安全区域！')

                        s = s_.copy()
                        dq.append(sum(env.s ** 2))
                        if done:
                            unsafe += 1
                            break
                            # print('=========进入非安全区域!=========')
                            # print(s_)
                            # print(unsafe)
                        else:
                            if (len(dq) == mxlen and np.var(dq) < 1e-10) or tot >= 1000:
                                safe += 1
                                break

            succ_rate = 0
            if safe > 0:
                succ_rate = safe / (safe + unsafe)
                print('safe' + str(safe) + '  unsafe' + str(unsafe))
            print('第{}次迭代采样{}个点:成功率{}%, 总采样点数{}'.format(t, safe + unsafe, succ_rate, tot_num))
            output0.append(t)
            output1.append(succ_rate)
            # ===========随机采样n个点求成功率================
            if (safe_time >= 50) or (end - start) / 60 / 60 > 1:
                # end = time.time()
                ti = (end - start)
                ti = round(ti, 2)
                print('测试集{}的训练用时：{}s'.format(i, ti))
                print('测试集{}的训练轮数：{}'.format(i, t))
                print('=====连续{}条轨迹安全，测试通过=======', format(env.k))
                agent.save_model()
                # draw(env,i,agent)
                # draw(env)
                testest(agent, env, i)
                print('层数:{}\n每层{}个结点\n{}激活函数\n'.format(env.dense, env.units, env.activation))
                print('训练耗时：' + str(ti) + 's' + '\n')
                print('训练轮数:' + str(t + 1) + '\n')
                break

        if not os.path.exists('save'):
            os.makedirs('save')
        with open('save/line_ex{}{}_meta_1.txt'.format(env.id, '_with_lidao' if env.is_lidao else ''), 'a',
                  encoding='utf-8') as f:
            for k in range(0, len(output0)):
                f.write(str(output0[k]) + ' ' + str(output1[k]) + '\n')


if __name__ == '__main__':
    # agent = train()
    # fit_poly(env, agent, ddpg)  # 拟合多项式
    #  test()
    # draw_line()
    # draw(get_Env(3))
    train_with_certain_times()
