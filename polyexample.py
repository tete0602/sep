import sys
sys.path.append("Meta_DDPG_v3")
from DQN import DeepQNetwork
from DDPG import DDPG
from Env2 import ex_12_dim
import pandas as pd
import numpy as np
import time
from collections import deque
import os



env=ex_12_dim(1)
ddpg = True
action_value = np.linspace(-env.u, env.u, 2)
parameter_test = './parameter_test/con2/dim_12.txt'


def train(index, env, ddpg, action_value, s_sum):

    # print('第{}个训练例子'.format(index))
    agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path='DDPG'+env.path, units=env.units, dense=env.dense,
                     activation=env.activation)
    # print('输出范围：', action_value)

    tot = 0
    mxlen = 50
    dq = deque(maxlen=mxlen)
    output0 = deque(maxlen=10005)
    output1 = deque(maxlen=10005)
    var = 3
    #---------------随机取100个初始点-----------
    # s_sum=[]
    epoch=1
    s_unsafe = []
    unsafe_time=0


    # for _ in range(100):
    #     s=env.reset()
    #     s_sum.append(s)
    print('初始点集合', s_sum)

    #------------------------------------------
    start = time.time()
    print('第{}个epoch ：'.format(epoch))
    for i in range(len(s_sum)):
        #
        # agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path='old' + env.path, units=env.units,
        #              dense=env.dense,
        #              activation=env.activation)
        print('第{}个初始点：'.format(i))
        s_start=s_sum[i]
        s = s_sum[i]
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
                unsafe_time += 1
                s_unsafe.append(s)
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
            # if done or (len(dq) == mxlen and np.var(dq) < 1e-5):
            # if done or (var < 1e-5):
            if done or (len(dq) == mxlen and np.var(dq) < 1e-10) or count == 300:
                print('reward:', reward, 'info:', info, (' Explore:', var))
                break
            s = s_.copy()

            # del agent
            # gc.collect()



    print('不安全初始点个数：'+str(unsafe_time)+'\n')
    print('不安全初始点列表：'+str(s_unsafe)+'\n')
    print('初始成功率：'+str((100-unsafe_time)/100)+'\n')
    # print('所需epoch数'.format(epoch))
    with open(parameter_test, 'a', encoding='utf-8') as f:
        f.write('第{}个训练例子'.format(index)+'\n')
        f.write('初始成功率：'+str((100-unsafe_time)/100)+'\n')
    # ss_unsafe = []
    a = time.time()
    while unsafe_time != 0:
        ss_unsafe = []

        epoch += 1
        print('-------'+'对不安全初始点继续下个epoch'+'-----')
        print('第{}个epoch'.format(epoch))
        with open(parameter_test, 'a', encoding='utf-8') as f:
            f.write('第{}个epoch'.format(epoch) + '\n')
        unsafe_time=0

        if epoch == 50:  # 当epoch=50时，则跳出循环
            print('已经达到最大epoch，退出，训练控制器失败')
            with open(parameter_test, 'a', encoding='utf-8') as f:

                f.write('控制器训练失败' + '\n')

            break
        # print(epoch)
        # agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path='old' + env.path, units=env.units,
        #              dense=env.dense,
        #              activation=env.activation)

        with open(parameter_test, 'a', encoding='utf-8') as f:
            f.write('不安全初始点共有'+str(len(s_unsafe))  + '\n')

        for j in range(len(s_unsafe)):

            print('第{}个不安全始点:'.format(j))

            s = s_unsafe[j]
            # print('初始点:', s_sum[i])
            reward = 0
            count = 0
            dq = deque(maxlen=mxlen)
            # s_unsafe.pop(j)
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
                    unsafe_time += 1
                    ss_unsafe.append(s)
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

                # if done or (len(dq) == mxlen and np.var(dq) < 1e-5):
                if done or (len(dq) == mxlen and np.var(dq) < 1e-2):

                    print('reward:', reward, 'info:', info, (' Explore:', var))
                    # s_unsafe-=s
                    break
                s = s_.copy()
        s_unsafe=ss_unsafe
    b=time.time()
    with open(parameter_test, 'a', encoding='utf-8') as f:
        f.write('此轮耗费时间为' + str(b-a) + '\n')


                # epoch += 1  # 每次循环后+1
    success=(100-unsafe_time) / 100
    print('最终成功率为 {:.2%}'.format((100-unsafe_time) / 100))
    print('总共需要{}个epoch'.format(epoch)+'\n')
    end = time.time()
    ddpg_cost=end-start
    print('测试参数所耗费时长'+str(ddpg_cost))
    with open(parameter_test, 'a', encoding='utf-8') as f:
        # f.write('第{}个例子'.format(index) +str(s_sum)+ '\n')
        # f.write('第{}个初始点为'.format(epoch) + '\n')
        f.write('总共需要{}个epoch'.format(epoch)+'\n')
        f.write('测试参数所耗费时长'+str(ddpg_cost)+'\n')
        f.write('-------------------------------------------'+ '\n')
        f.write('-------------------------------------------' + '\n')
# print('安全轨迹数：',safe_time)
# end = time.time()

# if (safe_time >= 50):
#     ti = (end - start)
#     ti = round(ti, 2)
#     print('训练用时：{}s 训练轮数'.format(ti//t))
#     agent.save_model()
#     #draw()
#    # testest(agent,env)
#
#     with open('cofe/50_message_9.txt', 'a', encoding='utf-8') as f:
#         #f.write('层数:{}\n每层{}个结点\n{}激活函数\n'.format(env.dense, env.units, env.activation))
#         f.write('第{}个env\n'.format(index))
#         f.write('安全轨迹数：'+str(safe_time)+'\n')
#         f.write('训练耗时：' + str(ti) + 's' + '\n')
#         f.write('训练轮数:' + str(t + 1) + '\n')
#         f.write('不安全域:' + str(env.U_zones) + '\n')
#         f.write('安全区域:' + str(env.I_zones) + '\n')
#         f.write('-----------------------------------\n')
#     break
# else:
#     if((end - start) / 60/60  > 1):
#         with open('cofe/50_message_9.txt', 'a', encoding='utf-8') as f:
#             f.write('第{}个env'.format(index)+'---训练失败'+'\n')
#             f.write('-----------------------------------\n')
#         break


    return index,ddpg_cost,epoch,success,s_start


def train_with_envlist_ddpg(envlist, tot):
    # filename = 'cofe/50_message_9.txt'
    filename = 'parameter_test/con2/dim_12.txt'
    filename1 = 'parameter_test/con2/50_mate_dim_9.csv'
    if os.path.exists(filename):
        os.remove(filename)
    dict_ddpg={}
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    for i, item in enumerate(envlist):
        env = item
        # env.s = tot[i][-1]
        env.s = np.array([np.random.random() - 0.5 for _ in range(12)])
        ddpg = True
        action_value = np.linspace(-env.u, env.u, 2)

        index,ddpg_cost,epoch,succsess,s_start=train(i, env, ddpg, action_value, tot[i])
        # dict_ddpg.update({'MAML第几个例子': index,'DDPG训练时长': ddpg_cost,  'MAML总共需要几个epoch': epoch})

        list1.append(index)
        list2.append(ddpg_cost)
        list3.append(epoch)
        list4.append(succsess)
        list5.append(s_start)
        # dict_ddpg['MAML第几个例子'] = index
        # dict_ddpg['DDPG训练时长'] = ddpg_cost
        # dict_ddpg['MAML总共需要几个epoch'] = epoch
    dict_ddpg={'DDPG第几个例子':list1,'DDPG训练时长':list2,'DDPG总共需要几个epoch':list3,'最终成功率':list4,'初始点':list5}

    df1 = pd.DataFrame(dict_ddpg)


    # 保存 dataframe


    df1.to_csv('G:\Meta_DDPG\parameter_test\con2\dim_12_50.csv')
    # lock.release()








if __name__ == '__main__':
    pass
    # train(1, env, ddpg, action_value)
    # train_line()
    #testest(agent,env)
    #draw()
    # fit_poly(env, agent, ddpg)  # 拟合多项式
    #test(model='save/meta/olddim9_2_test/model')
    # train_with_envs()
