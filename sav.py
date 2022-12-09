# mamlRL的对比实验
# 使用原来方式进行训练
# 验证条件是，连续50条轨迹不进入非安全区域则标志控制器生成成功
# 暂不考虑多项式拟合和验证的问题

from DQN import DeepQNetwork
from DDPG import DDPG
from Env import get_Env
from Env import get_testEnv
import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from matplotlib import pyplot as plt
from fit_poly import fit_poly
import time
from collections import deque
import os

env = get_testEnv(0)
ddpg = True
action_value = np.linspace(-env.u, env.u, 2)


def train():
    if ddpg:
        agent = DDPG(1, a_bound=env.u, s_dim=env.n_obs, is_train=True, path=env.path, units=env.units, dense=env.dense,
                     activation=env.activation)
    else:
        agent = DeepQNetwork(n_action=2, n_state=env.n_obs, path=env.path, is_train=True, dense=env.dense,
                             activation=env.activation)

    print('输出范围：', action_value)
    tot = 0
    start = time.time()

    safe_time = 0
    mxlen = 50
    dq = deque(maxlen=mxlen)

    var = 3
    for t in range(100000):
        s = env.reset()
        #print('初始点:', s)
        reward = 0

        count = 0
        while True:
            count += 1
            dq.append(sum(env.s ** 2))

            a = agent.choose_action(s)
            if ddpg:
                a_v = np.clip(np.random.normal(a, var), -env.u, env.u)[
                    0]  # add randomness to action selection for exploration
            # if tot%1000:
            #     print(a)
            else:
                a_v = np.dot(a, action_value.T)

            # print(action_value[np.argmax(a)],a_v)
            s_, r, done, info = env.step(a_v)
            if info[1]:
                print('李导数不满足!!')
            if not ddpg:
                a_v = np.argmax(a)
            agent.store_transition(s, a_v, r, s_, done)
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
                # 学习
                agent.learn()

            # np.var variance均方误差，axis=0 按列求方差，=1按行求方差
            if done or (len(dq) == mxlen and np.var(dq) < 1e-10):
                print('reward:', reward, 'info:', info, (' Explore:', var) if ddpg else ('e_greedy:', agent.e_greedy))

                break
            s = s_.copy()
        safe_time += 1

        end = time.time()
        print('安全轨迹数：',safe_time)
        if (safe_time > env.k) or (end - start) / 60 / 60 > 1:
            ti = (end - start)
            ti = round(ti, 2)
            print('testEnv0,训练用时：{}s'.format(ti))
            if not os.path.exists('./cofe'):
                os.makedirs('./cofe')
            with open('./cofe/message_ex{}{}.txt'.format(env.id, '_with_lidao' if env.is_lidao else ''), 'w',
                      encoding='utf-8') as f:
                f.write('层数:{}\n每层{}个结点\n{}激活函数\n'.format(env.dense, env.units, env.activation))
                f.write('训练耗时：' + str(ti) + 's' + '\n')
                f.write('训练轮数:' + str(t + 1) + '\n')
            break

    return agent


def test():
    X1 = []
    X2 = []
    model = joblib.load('model/ex{}{}.model'.format(env.id, '_with_lidao' if env.is_lidao else ''))
    s = env.reset()
    P = PolynomialFeatures(env.degree, include_bias=False)
    tot = 0
    print('初始状态:', s)
    mxlen = 50
    dq = deque(maxlen=mxlen)
    while True:
        tot += 1

        # a = agent.choose_action(s)
        # a_v = np.dot(a, action_value.T)
        X1.append(s[0])
        X2.append(s[1])

        s = P.fit_transform([s])
        a_v = model.predict(s)[0]
        print(a_v)
        s_, r, done, info = env.step(a_v)
        if info[1] == True:
            print('李导数不满足！')
        s = s_.copy()
        dq.append(sum(env.s ** 2))
        if len(dq) == mxlen and np.var(dq) < 1e-10:
            break
        if tot > 4000:
            break
        if done:
            print('进入非安全区域!')
            print(s_)
            break

    if env.D_zones.shape == 'box':
        plt.xlim(env.D_zones.low[0], env.D_zones.up[0])
        plt.ylim(env.D_zones.low[1], env.D_zones.up[1])
    else:
        r = env.D_zones.r
        plt.xlim(env.D_zones.center[0] - r, env.D_zones.center[0] + r)
        plt.ylim(env.D_zones.center[1] - r, env.D_zones.center[1] + r)
        thta = np.linspace(0, 2 * np.pi, 100)
        x = [r * np.cos(v) + env.D_zones.center[0] for v in thta]
        y = [r * np.sin(v) + env.D_zones.center[1] for v in thta]
        plt.plot(x, y)

    print('轨迹长度:', len(X1))
    plt.plot(X1, X2)

    if env.I_zones.shape == 'ball':
        r = env.I_zones.r
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
        r = env.U_zones.r
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
    plt.show()


if __name__ == '__main__':
    agent = train()
    #fit_poly(env, agent, ddpg)  # 拟合多项式
    #test()
