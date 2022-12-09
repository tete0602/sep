#encoding=utf-8
import numpy as np
from sympy import *
import math
import random
# import gym
# e=gym.make('InvertedPendulum-v2')
# e.step(0.1)
# e.reset()
pi = math.pi
class Zones():  ## 定义一个区域 ：有两种，长方体或者球
    def __init__(self, shape, center=None, r=0.0, low=None, up=None):
        self.shape = shape
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r  ##半径的平方
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2  ## 外接球中心
            self.r = sum(((self.up - self.low) / 2) ** 2)  ## 外接球半径平方
        else:
            raise ValueError('没有形状为{}的区域'.format(shape))


class Example():
    def __init__(self, n_obs, D_zones, I_zones, U_zones, f, u, degree, path, dense, units, activation, id, k, B=None):
        self.n_obs = n_obs  # 变量个数
        self.D_zones = D_zones  # 不变式区域
        self.I_zones = I_zones  ## 初始区域
        self.U_zones = U_zones  ## 非安全区域
        self.f = f  # 微分方程
        self.B = B  # 障碍函数
        self.u = u  # 输出范围为 [-u,u]
        self.degree = degree  # 拟合该多项式的次数
        self.path = path  # 存储该例子RL参数的路径
        self.dense = dense  # 网络的层数
        self.units = units  # 每层节点数
        self.activation = activation  # 激活函数
        self.k = k  # 提前停止的轨迹条数
        self.id = id  # 标识符


class Env():
    def __init__(self, example):
        self.n_obs = example.n_obs
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.U_zones = example.U_zones
        self.f = example.f
        self.B = example.B
        self.path = example.path
        self.u = example.u
        self.degree = example.degree
        self.dense = example.dense  # 网络的层数
        self.units = example.units  # 每层节点数
        self.activation = example.activation  # 激活函数
        self.id = example.id
        self.dt = 0.01 #步长
        self.k = example.k
        self.is_lidao = False if self.B == None else True
        if self.is_lidao:
            self.path = self.path.split('/')[0] + '_with_lidao' + '/' + self.path.split('/')[1]
            print(self.path)

    def unisample(self,s):
        self.s = s

    def reset(self):
        self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])  ##边长为1，中心在原点的正方体的内部，产生-0.5~0.5的随机数，组成
        if self.I_zones.shape == 'ball':
            ## 在超球内进行采样：将正方体进行归一化，变成对单位球的表面采样，再对其半径进行采样。
            self.s *= 2  ## 变成半径为1
            self.s = self.s / np.sqrt(sum(self.s ** 2)) * self.I_zones.r * np.random.random() ** (
                    1 / self.n_obs)  ##此时球心在原点
            ## random()^(1/d) 是为了均匀采样d维球体
            self.s += self.I_zones.center

        else:
            self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        return self.s

    def step(self, u):
        self.ds = np.array([F(self.s, u) for F in self.f])
        self.s = self.s + self.ds * self.dt

        if self.D_zones.shape == 'box':
            self.s = np.array(
                [min(max(self.s[i], self.D_zones.low[i]), self.D_zones.up[i]) for i in range(self.n_obs)]
            )

        else:
            t = np.sqrt(self.D_zones.r / sum(self.s ** 2))
            if t < 1:
                self.s = self.s * t

        if self.U_zones.shape == 'ball':
            is_unsafe = sum((self.s - self.U_zones.center) ** 2) < self.U_zones.r
        else:
            safe = 0
            for i in range(self.n_obs):
                if self.U_zones.low[i] <= self.s[i] <= self.U_zones.up[i]:
                    safe = safe + 1
            is_unsafe = (safe == self.n_obs)

        dis = sum((self.s - self.U_zones.center) ** 2) - self.U_zones.r
        reward = dis / 4

        #if is_unsafe:
         #   reward = -reward * 4
        li_dao = False
        if self.is_lidao:
            if not self.get_sign(u):
                reward -= self.D_zones.r / 4
                li_dao = True

        return self.s, reward, is_unsafe, (is_unsafe, li_dao)

    def get_sign(self, u):
        sb = ['x' + str(i) for i in range(self.n_obs)]
        x = symbols(sb)  # 求导用
        B = self.B(x)  # 障碍函数
        x_0 = {k: v for k, v in zip(sb, self.s)}
        if B.subs(x_0) >= 0.1:
            return True
        su = sum([diff(B, x[i]).subs(x_0) * self.f[i](self.s, u) for i in range(self.n_obs)])
        return su > 0








def ex_9_dim(i,a,b):
        # 实际就1个例子
    examples = {
            0:Example(#----原本例子-----#----\cite{chen2020novelchen2020novel}
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.99] * 9, up=[1.01] * 9),
            U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8]
               ],
            B=None,
            u=3,
            degree=2,
            path='dim9_0_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=100  # 3000
        )

    }
    for key in range(1,50):
        random_a = 0 + (0.99 - 0) * np.random.random()
        random_b = np.random.random()
        examples.update({
            key: Example(  # ----原本例子-----#----\cite{chen2020novelchen2020novel}
                n_obs=9,
                D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
                I_zones=Zones('box', low=[0.99] * 9, up=[1.01] * 9),
                U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
                f=[lambda x, u: 3 * x[2] + u,
                   lambda x, u: x[3] - x[1] * x[5],
                   lambda x, u: x[0] * x[5] - 3 * x[2],
                   lambda x, u: x[1] * x[5] - x[3],
                   lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
                   lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
                   lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
                   lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
                   lambda x, u: 2 * x[5] * x[7] - x[8]
                   ],
                B=None,
                u=3,
                degree=2,
                path='dim9_0_test/model',
                dense=5,
                units=30,
                activation='relu',
                id=12,
                k=100  # 3000
            )
        })

    return examples




# if __name__ == '__main__':
#     # a=random.random()
#     # b=random.random()
#     # i=1
#     # for _ in range(10):
#     #     dict_9=ex_9_dim(i,a,b)
#     random_a=0 + (0.99-0)*np.random.random()
#     random_b = np.random.random()
#
#     dict = ex_9_dim(0, random_a, random_b)
#     for i in range(1,50):
#
#
#
#         dict.update(ex_9_dim(i, random_a, random_b))
#
#     def dim_9(j):
#
#         return Env(dict[j])





    # for i in a.keys():
    #     print(i)




