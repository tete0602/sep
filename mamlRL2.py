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
import pandas as pd

# from Env2 import ex_9_dim
from Env2 import ex_12_dim
# from Env2 import ex_6_dim
# from Env2 import ex_4_dim
# from Env2 import ex_2_dim
from Env import  uni_12dim_train

ddpg = True

# tasks_num = 2

# tasks_num_test = 2
train_num = 2
test_num = 2

epochs = 2
trajectory_num = 50
step_num = 300
# epochs=10
mate_parameter_test = './parameter_test/con2/mate_dim_12.txt'
# mate_parameter_test = './parameter_test/dac/mate_dim_12.txt'



def train():


    # global safe_time
    # --------------------------------------------------------------------------------------------------

    # 随机从数据集中选取一个system
    env_meta = ex_12_dim(random.randint(0, 49))
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
    # outputText = './outputText/50_mate_message_9.txt'

    # meta train
    #总的训练时长
    start_tot = time.time()

    for t in range(epochs):
        print('===========mate参数更新的第{}个epoch==========='.format(t))
        print('===========使用训练资料===========')
        #--------构造训练任务集---------------
        for i in range(train_num):
            # env = ex_12_dim(random.randint(0, 49))
            env=uni_12dim_train(i)
            agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
                         units=env.units, dense=env.dense, activation=env.activation)
            env_train.append(env)
            models.append(agent)
            # print(env_train)
            # print(models)
        # -----------------------------------

        # 对于每一个训练任务
        for i in range(train_num):
            # 将初始元网络模型参数赋值给第一个训练任务
            # models[i].assign(model_meta.get_apra())
            out1, out2, out3, out4 = model_meta.get_apra()
            models[i].assign(out1, out2, out3, out4)
            env = env_train[i]
            # print('输出范围：', action_value)
            print('==========train：第', i, '个训练example===========')
            # ---------------随机取100个初始点-----------
            mate_train_s_sum=[]
            mate_train_epoch = 1
            mate_train_s_unsafe = []
            mate_train_unsafe_time = 0

            for _ in range(100):
                mate_train_s = env.reset()
                mate_train_s_sum.append(mate_train_s)
            print('第{}个训练任务的初始点集合'.format(i), mate_train_s_sum)

            # ------------------------------------------
            print('第{}个训练任务的第{}个epoch ：'.format(i, mate_train_epoch))

            for j in range(len(mate_train_s_sum)):

                print('第{}个初始点：'.format(j))
                # models[i].assign(out1, out2, out3, out4)
                s = mate_train_s_sum[j]
                # print('初始点:', s_sum[i])
                reward = 0
                count = 0
                dq = deque(maxlen=mxlen)
                while True:
                    count += 1
                    dq.append(sum(env.s ** 2))

                    a = models[i].choose_action(s)

                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                    # add randomness to action selection for exploration
                    # if tot%1000:
                    #     print(a)

                    # print(action_value[np.argmax(a)],a_v)
                    s_, r, done, info = env.step(a_v)
                    if info[1]:
                        print('李导数不满足!!')

                    models[i].store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('不安全')
                        # 如果出现不安全轨迹就重新计数安全轨迹数量(保证连续安全)
                        # safe_time = 0
                        mate_train_unsafe_time += 1
                        mate_train_s_unsafe.append(s)
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
                    # if done or (len(dq) == mxlen and np.var(dq) < 1e-8):
                    #     print('reward:', reward, 'info:', info, (' Explore:', var))
                    #     break
                    if done or (len(dq) == mxlen and np.var(dq) < 1e-2):
                        print('reward:', reward, 'info:', info, (' Explore:', var))
                        break
                    s = s_.copy()

            print('不安全初始点个数：' + str(mate_train_unsafe_time) + '\n')
            print('不安全初始点列表：' + str(mate_train_s_unsafe) + '\n')
            print('初始成功率：' + str((100 - mate_train_unsafe_time) / 100) + '\n')


            while mate_train_unsafe_time != 0:
                mate_train_epoch += 1
                print('-------' + '对不安全初始点继续下个epoch' + '-----')
                print('第{}个epoch'.format(mate_train_epoch))
                mate_train_unsafe_time = 0

                if mate_train_epoch == 50:  # 当epoch=50时，则跳出循环
                    print('已经达到最大epoch，退出')
                    break
                print(mate_train_epoch)
                # models[i].assign(out1, out2, out3, out4)
                for k in range(len(mate_train_s_unsafe)):
                    print('第{}个不安全初始点:'.format(k))
                    s = mate_train_s_unsafe[k]
                    # print('初始点:', s_sum[i])
                    reward = 0
                    count = 0
                    dq = deque(maxlen=mxlen)
                    while True:
                        count += 1
                        dq.append(sum(env.s ** 2))

                        a = models[i].choose_action(s)

                        a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                        # add randomness to action selection for exploration
                        # if tot%1000:
                        #     print(a)

                        # print(action_value[np.argmax(a)],a_v)
                        s_, r, done, info = env.step(a_v)
                        if info[1]:
                            print('李导数不满足!!')

                        models[i].store_transition(s, a_v, r, s_, done)
                        reward += r
                        if done:
                            print('不安全')
                            # 如果出现不安全轨迹就重新计数安全轨迹数量(保证连续安全)
                            # safe_time = 0
                            mate_train_unsafe_time += 1
                            # mate_train_s_unsafe.append(s)
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
                        # if done or (len(dq) == mxlen and np.var(dq) < 1e-8):
                        #     print('reward:', reward, 'info:', info, (' Explore:', var))
                        #     break

                        if done or (len(dq) == mxlen and np.var(dq) < 1e-2):
                            print('reward:', reward, 'info:', info, (' Explore:', var))
                            break
                        s = s_.copy()
                # mate_train_epoch += 1  # 每次循环后+1
            print('总共需要{}个epoch'.format(mate_train_epoch) + '\n')

            # 对于每一条轨迹

            print('===========开始元更新===========epoch：', t)
            print('===========使用测试资料===========')
            testTask = random.randint(0, 49)
            env_test= ex_12_dim(testTask)
            # s = env.reset()
            # apra = models[i].get_apra()
            # meta_grads = []
            print('===========使用模型' + str(i) + '测试任务' + str(testTask) + '============')
            reward = 0
            agent_test = DDPG(1, a_bound=env_test.u, s_dim=env_test.n_obs, is_train=True, path=env_test.path,
                         units=env_test.units, dense=env_test.dense, activation=env_test.activation)

            out1, out2, out3, out4 = models[i].get_apra()
            agent_test.assign(out1, out2, out3, out4)

            # start = time.time()
            # ---------------随机取100个初始点-----------
            mate_test_s_sum=[]
            mate_test_epoch = 1
            mate_test_s_unsafe = []
            mate_test_unsafe_time = 0

            for _ in range(100):
                mate_test_s = env_test.reset()
                mate_test_s_sum.append(mate_test_s)
            print('第{}个测试任务的初始点集合'.format(i), mate_test_s_sum)

            # ------------------------------------------
            print('第{}个测试任务的第{}个epoch ：'.format(i, mate_test_epoch))

            for m in range(len(mate_test_s_sum)):
                # agent_test = DDPG(1, a_bound=env_test.u, s_dim=env_test.n_obs, is_train=True, path=env_test.path,
                #                   units=env_test.units, dense=env_test.dense, activation=env_test.activation)

                # out1, out2, out3, out4 = models[i].get_apra()
                # agent_test.assign(out1, out2, out3, out4)
                print('第{}个初始点：'.format(m))
                s = mate_test_s_sum[m]
                # print('初始点:', s_sum[i])
                reward = 0
                count = 0
                dq = deque(maxlen=mxlen)
                while True:
                    count += 1
                    dq.append(sum(env.s ** 2))

                    a = agent_test.choose_action(s)

                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                    # add randomness to action selection for exploration
                    # if tot%1000:
                    #     print(a)

                    # print(action_value[np.argmax(a)],a_v)
                    s_, r, done, info = env.step(a_v)
                    if info[1]:
                        print('李导数不满足!!')

                    model_meta.store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('不安全')
                        # 如果出现不安全轨迹就重新计数安全轨迹数量(保证连续安全)
                        # safe_time = 0
                        mate_test_unsafe_time += 1
                        mate_test_s_unsafe.append(s)
                        tot = tot // 2000 * 2000 + 1010

                    if tot % 2000 == 0:
                        done = 1
                    tot += 1
                    if tot > 1000:
                        if tot % 10 == 0:
                            var *= .9995
                        # 学习
                        model_meta.learn()

                    # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                    # if done or (len(dq) == mxlen and np.var(dq) < 1e-8):
                    #     print('reward:', reward, 'info:', info, (' Explore:', var))
                    #     break
                    if done or (len(dq) == mxlen and np.var(dq) < 1e-2):
                        print('reward:', reward, 'info:', info, (' Explore:', var))
                        break
                    s = s_.copy()

            print('不安全初始点个数：' + str(mate_test_unsafe_time) + '\n')
            print('不安全初始点列表：' + str(mate_test_s_unsafe) + '\n')
            print('初始成功率：' + str((100 - mate_test_unsafe_time) / 100) + '\n')
            # print('所需epoch数'.format(epoch))
            # with open('parameter_test/mate_dim_2.txt', 'a', encoding='utf-8') as f:
            #     f.write('第{}个训练例子'.format(index) + '\n')
            #     f.write('初始成功率：' + str((100 - mate_test_unsafe_time) / 100) + '\n')

            while mate_test_unsafe_time != 0:
                mate_test_epoch+= 1
                print('-------' + '对不安全初始点继续下个epoch' + '-----')
                print('第{}个epoch'.format(mate_test_epoch))
                mate_test_unsafe_time = 0

                if mate_test_epoch == 50:  # 当epoch=50时，则跳出循环
                    print('已经达到最大epoch，退出')
                    break
                print(mate_test_epoch)
                # agent_test = DDPG(1, a_bound=env_test.u, s_dim=env_test.n_obs, is_train=True, path=env_test.path,
                #                   units=env_test.units, dense=env_test.dense, activation=env_test.activation)
                # out1, out2, out3, out4 = models[i].get_apra()
                # agent_test.assign(out1, out2, out3, out4)
                for n in range(len(mate_test_s_unsafe)):
                    print('第{}个不安全初始点'.format(n))
                    s = mate_test_s_unsafe[n]
                    # print('初始点:', s_sum[i])
                    reward = 0
                    count = 0
                    dq = deque(maxlen=mxlen)
                    while True:
                        count += 1
                        dq.append(sum(env.s ** 2))

                        a = agent_test.choose_action(s)

                        a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                        # add randomness to action selection for exploration
                        # if tot%1000:
                        #     print(a)

                        # print(action_value[np.argmax(a)],a_v)
                        s_, r, done, info = env.step(a_v)
                        if info[1]:
                            print('李导数不满足!!')

                        model_meta.store_transition(s, a_v, r, s_, done)
                        reward += r
                        if done:
                            print('不安全')
                            # 如果出现不安全轨迹就重新计数安全轨迹数量(保证连续安全)
                            # safe_time = 0
                            mate_test_unsafe_time += 1
                            # mate_test_s_unsafe.append(s)
                            tot = tot // 2000 * 2000 + 1010

                        if tot % 2000 == 0:
                            done = 1
                        tot += 1
                        if tot > 1000:
                            if tot % 10 == 0:
                                var *= .9995
                            # 学习
                            model_meta.learn()

                        # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                        # if done or (len(dq) == mxlen and np.var(dq) < 1e-8):
                        #     print('reward:', reward, 'info:', info, (' Explore:', var))
                        #     break

                        if done or (len(dq) == mxlen and np.var(dq) < 1e-2):
                            print('reward:', reward, 'info:', info, (' Explore:', var))
                            break
                        s = s_.copy()
                # mate_test_epoch += 1  # 每次循环后+1
            print('总共需要{}个epoch'.format(mate_test_epoch) + '\n')
            # end = time.time()
            # print('训练参数所耗费时长' + str(end - start))
            # with open('parameter_test/dim_2.txt', 'a', encoding='utf-8') as f:
            #     # f.write('总共需要{}个epoch'.format(epoch) + '\n')
            #     f.write('训练参数所耗费时长' + str(end - start) + '\n')

    model_meta.save_model()
    end_tot = time.time()
    tot_cost=end_tot-start_tot
    # print('总训练参数用时：{}s'.format(tot_cost))
    with open(mate_parameter_test, 'a', encoding='utf-8') as f:
        # f.write('第{}个env'.format(index)+'\n')
        # f.write('总训练参数用时：{}s'.format(tot_cost) + '\n')
        f.write('总训练参数用时：' + str(tot_cost) + '\n')



    return model_meta,tot_cost


# def train_with_certain_times():
#     os.remove('./outputText/mate_message_12.txt')
#     for i in range(50):
#         train(i)
#         test(i)

def train_with_envlist_maml(envlist,tot):
    # filename = './outputText/50_mate_message_9.txt'
    filename = 'parameter_test/con2/mate_dim_12.txt'
    if os.path.exists(filename):
        os.remove(filename)

    model_meta,tot_cost = train()
    # nme1 = ["MAML训练参数用时"]
    # 字典
    dict_maml = {}
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for i, item in enumerate(envlist):
        env = item
        # env.s = tot[i][-1]
        env.s = np.array([np.random.random() - 0.5 for _ in range(12)])


        index,mate_epoch,tot_cost_test,success,mate_s_start=test(i, item, model_meta,tot[i])

    # nme2 = ["MAML第几个例子"]
    # nme3 = ["MAML总共需要几个epoch"]
    # nme4 = ["MAML训练时长"]
        list1.append(index)
        list2.append(mate_epoch)
        list3.append(tot_cost_test)
        list4.append(success)
        list5.append(mate_s_start)

        # nme = ["MAML训练参数用时", "MAML第几个例子", "MAML总共需要几个epoch", "MAML训练时长"]
        # st = [tot_cost, index, mate_epoch, tot_cost_test]

        # dict_maml.update({'MAML第几个例子': index,'MAML训练参数用时': tot_cost,  'MAML总共需要几个epoch': mate_epoch, 'MAML训练时长': tot_cost_test})
    dict_maml = {'MAML第几个例子': list1, 'MAML参数训练时长': tot_cost, 'MAML总共需要几个epoch': list2,'MAML训练时长': list3, '最终成功率':list4,'初始点':list5}

    df = pd.DataFrame(dict_maml)


    # 保存 dataframe
    df.to_csv('G:\Meta_DDPG\parameter_test\con2\mate_dim_12_50.csv')





    # lock.release();


def test(index, env_test, model_meta,mate_s_sum):
    # model = joblib.load('model/ex{}{}.model'.format(env.id, '_with_lidao' if env.is_lidao else ''))
    # env_meta = get_singleTest(1)
    # env_mate = ex_12_dim(index)
    # model_meta = DDPG(1, a_bound=env_meta.u, s_dim=env_meta.n_obs, is_train=False, path=env_meta.path,
    #                   units=env_meta.units, dense=env_meta.dense, activation=env_meta.activation)
    # print(mate_s_sum)
    model_meta=model_meta
    task_test_num = 1
    start_tot=time.time()
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

        out1, out2, out3, out4 = model_meta.get_apra()
        agent.assign(out1, out2, out3, out4)

        tot = 0
        start = time.time()
        safe_time = 0
        mxlen = 50
        dq = deque(maxlen=mxlen)
        var = 3
        action_value = np.linspace(-env.u, env.u, 2)
        print('输出范围：', action_value)
        # out1, out2, out3, out4 = model_meta.get_apra()
        # agent.assign(out1, out2, out3, out4)
        # ---------------随机取100个初始点-----------
        # s_sum=[]
        mate_epoch = 1
        mate_s_unsafe = []
        mate_unsafe_time = 0
        safe_time = 0

        # for _ in range(100):
        #     s=env.reset()
        #     s_sum.append(s)
        print('初始点集合', mate_s_sum)
        # print('初始点集合长度', len(s_sum))
        # ------------------------------------------
        start = time.time()
        print('第{}个epoch ：'.format(mate_epoch))
        for i in range(len(mate_s_sum)):
            # agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
            #              units=env.units, dense=env.dense, activation=env.activation)
            # out1, out2, out3, out4 = model_meta.get_apra()
            # agent.assign(out1, out2, out3, out4)
            print('第{}个初始点：'.format(i))
            mate_s_start =mate_s_sum[i]
            s= mate_s_sum[i]
            # print('初始点:', s_sum[i])
            reward = 0
            count = 0
            dq = deque(maxlen=mxlen)
            while True:
                count += 1
                dq.append(sum(env.s ** 2))

                a = agent.choose_action(s)

                a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                # add randomness to action selection for exploration
                # if tot%1000:
                #     print(a)

                # print(action_value[np.argmax(a)],a_v)
                s_, r, done, info = env.step(a_v)
                if info[1]:
                    print('李导数不满足!!')

                agent.store_transition(s, a_v, r, s_, done)
                reward += r
                if done:
                    print('不安全')
                    # 如果出现不安全轨迹就重新计数安全轨迹数量(保证连续安全)
                    # safe_time = 0
                    mate_unsafe_time += 1
                    mate_s_unsafe.append(s)
                    tot = tot // 2000 * 2000 + 1010

                if tot % 2000 == 0:
                    done = 1
                tot += 1
                if tot > 1000:
                    if tot % 10 == 0:
                        var *= .9995
                    # 学习
                    agent.learn()

                # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                # if done or (len(dq) == mxlen and np.var(dq) < 1e-8):
                #     print('reward:', reward, 'info:', info, (' Explore:', var))
                #     break
                if done or (len(dq) == mxlen and np.var(dq) < 1e-2):
                    print('reward:', reward, 'info:', info, (' Explore:', var))
                    break
                s = s_.copy()

        print('不安全初始点个数：' + str(mate_unsafe_time) + '\n')
        print('不安全初始点列表：' + str(mate_s_unsafe) + '\n')
        print('初始成功率：' + str((100 - mate_unsafe_time) / 100) + '\n')
        # print('所需epoch数'.format(epoch))
        # with open('parameter_test/mate_dim_9.txt', 'a', encoding='utf-8') as f:
        #     f.write('第{}个训练例子'.format(index) + '\n')
        #     f.write('初始成功率：' + str((100 - mate_unsafe_time) / 100) + '\n')
        # mate_ss_unsafe=[]
        a = time.time()
        while mate_unsafe_time != 0:
            mate_ss_unsafe = []
            mate_epoch += 1
            print('-------' + '对不安全初始点继续下个epoch' + '-----')
            print('第{}个epoch'.format(mate_epoch))
            mate_unsafe_time = 0
            with open(mate_parameter_test, 'a', encoding='utf-8') as f:
                f.write('第{}个epoch'.format(mate_epoch)+ '\n')


            if mate_epoch == 50:  # 当epoch=50时，则跳出循环
                print('已经达到最大epoch，退出，训练控制器失败')
                with open(mate_parameter_test, 'a', encoding='utf-8') as f:
                    # f.write('第{}个env'.format(index)+'\n')
                    f.write('控制器训练失败' + '\n')

                break
            print(mate_epoch)
            # agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path,
            #              units=env.units, dense=env.dense, activation=env.activation)
            # out1, out2, out3, out4 = model_meta.get_apra()
            # agent.assign(out1, out2, out3, out4)
            print('共{}个不安全初始点'.format(str(len(mate_s_unsafe))) + '\n')
            with open(mate_parameter_test, 'a', encoding='utf-8') as f:
                f.write('不安全初始点共有'+str(len(mate_s_unsafe)) + '\n')

            for j in range(len(mate_s_unsafe)):
                print('第{}个不安全初始点'.format(j))


                s = mate_s_unsafe[j]
                # print('初始点:', s_sum[i])
                reward = 0
                count = 0
                dq = deque(maxlen=mxlen)
                while True:
                    count += 1
                    dq.append(sum(env.s ** 2))

                    a = agent.choose_action(s)

                    a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[0]
                    # add randomness to action selection for exploration
                    # if tot%1000:
                    #     print(a)

                    # print(action_value[np.argmax(a)],a_v)
                    s_, r, done, info = env.step(a_v)
                    if info[1]:
                        print('李导数不满足!!')

                    agent.store_transition(s, a_v, r, s_, done)
                    reward += r
                    if done:
                        print('不安全')
                        # 如果出现不安全轨迹就重新计数安全轨迹数量(保证连续安全)
                        # safe_time = 0
                        mate_unsafe_time += 1
                        mate_ss_unsafe.append(s)
                        tot = tot // 2000 * 2000 + 1010

                    if tot % 2000 == 0:
                        done = 1
                    tot += 1
                    if tot > 1000:
                        if tot % 10 == 0:
                            var *= .9995
                        # 学习
                        agent.learn()

                    # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
                    # if done or (len(dq) == mxlen and np.var(dq) < 1e-8):
                    #     print('reward:', reward, 'info:', info, (' Explore:', var))
                    #     break

                    if done or (len(dq) == mxlen and np.var(dq) < 1e-2):
                        print('reward:', reward, 'info:', info, (' Explore:', var))
                        break
                    s = s_.copy()
            mate_s_unsafe=mate_ss_unsafe

        b = time.time()
        with open(mate_parameter_test, 'a', encoding='utf-8') as f:
            f.write('此轮耗费时间为' + str(b - a) + '\n')
            # mate_epoch += 1  # 每次循环后+1
        print('总共需要{}个epoch'.format(mate_epoch) + '\n')
        success = (100 - mate_unsafe_time) / 100
        print('最终成功率为 {:.2%}'.format(success))
    end_tot = time.time()
    tot_cost_test=end_tot-start_tot
    print('测试参数所耗费时长' + str(tot_cost_test))

    with open(mate_parameter_test, 'a', encoding='utf-8') as f:
        f.write('第{}个例子'.format(index) + '\n')
        f.write('总共需要{}个epoch'.format(mate_epoch) + '\n')
        f.write('用meta参数训练新任务耗费时长' + str(tot_cost_test) + '\n')

    return index,mate_epoch,tot_cost_test,success,mate_s_start









if __name__ == '__main__':
    pass
    # agent = train()
    # fit_poly(env, agent, ddpg)  # 拟合多项式
    #  test()
    # draw_line()
    # draw(get_Env(3))
    # train_with_certain_times()
