# encoding=utf-8
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
            self.low = np.array(low)
            self.up = np.array(up)

            self.center = np.array(center)
            self.r = r  ##半径的平方
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2  ## 外接球中心
            self.r = sum(((self.up - self.low) / 2) ** 2)  ## 外接球半径平方
        else:
            raise ValueError('没有形状为{}的区域'.format(shape))

    def __str__(self):
        return 'Zones:{' + 'shape:{}, center:{}, r:{}, low:{}, up:{}'.format(self.shape, self.center, self.r, self.low,
                                                                             self.up) + '}'


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
        self.dt = 0.01  # 步长
        self.k = example.k
        self.is_lidao = False if self.B == None else True
        if self.is_lidao:
            self.path = self.path.split('/')[0] + '_with_lidao' + '/' + self.path.split('/')[1]
            print(self.path)

    def unisample(self, s):
        self.s = s

    # def reset(self):
    #     self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])  ##边长为1，中心在原点的正方体的内部，产生-0.5~0.5的随机数，组成
    #     if self.I_zones.shape == 'ball':
    #         ## 在超球内进行采样：将正方体进行归一化，变成对单位球的表面采样，再对其半径进行采样。
    #         self.s *= 2  ## 变成半径为1
    #         self.s = self.s / np.sqrt(sum(self.s ** 2)) * self.I_zones.r * np.random.random() ** (
    #                 1 / self.n_obs)  ##此时球心在原点
    #         ## random()^(1/d) 是为了均匀采样d维球体
    #         self.s += self.I_zones.center
    #
    #     else:
    #         self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
    #     return self.s
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
            # self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
            self.s = random.uniform(self.I_zones.up , self.I_zones.low)

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

        # if is_unsafe:
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


def ex_2_dim(i):
    # 实际就5个例子
    examples = {
        0: Example(  ## 当前例子为展示用例----\cite{prajna2005optimization}
            n_obs=2,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[1.3, -0.1], up=[1.35, 0]),
            U_zones=Zones('box', low=[-2, -2], up=[-1.9, -1.9]),

            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u
               ],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            B=None,
            u=1,
            degree=1,
            path='textExp0/model',
            dense=5,
            units=30,
            activation='relu',
            id=0,
            k=50
        ),
        1: Example(  # ----\cite{aylward2008stability}
            n_obs=2,
            D_zones=Zones('ball', center=[0, 0], r=4),
            I_zones=Zones('box', low=[1, -0.2], up=[1.4, 0.2]),
            U_zones=Zones('box', low=[-2, -0.2], up=[-1.5, 0.2]),
            f=[lambda x, u: u - 0.5 * x[0] ** 3,
               lambda x, u: 3 * x[0] - x[1],
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='testExp1/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50,
        ),
        2: Example(  # ---\cite{sassi2014iterative}
            n_obs=2,
            D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
            I_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
            U_zones=Zones('ball', center=[-2.25, -1.75], r=0.25),

            f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u,
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7),
               ],
            B=None,  ## 没有障碍函数写 None
            u=0.3,
            degree=3,
            path='testExp2/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        3: Example(  # --------------\cite{bouissou2014computation}
            n_obs=2,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[-1.5, -1.5], up=[-1.4, -1.3]),
            U_zones=Zones('box', low=[-0.1, 0.5], up=[0.1, 1]),
            f=[lambda x, u: -x[0] + x[0] * x[1],
               lambda x, u: u - x[0] + 0.25 * x[1],
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=2,
            path='testExp3/model',
            dense=5,
            units=30,
            activation='relu',
            id=3,
            k=50,
        ),
        4: Example(  # 初始参数1 成功 #初始参数2 失败 # 参数3 前期失败，后期成功------\cite{prajna2004nonlinear}
            n_obs=2,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('ball', center=[1, 0], r=0.1),
            U_zones=Zones('ball', center=[-1, 1], r=2.5),
            f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='trainExp1/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=50,
        )
    }

    a = random.random()
    b = random.random()*2
    D_zones_list = [Zones('box', low=[-2, -2], up=[2, 2]),
                    Zones('ball', center=[0, 0], r=4),
                    Zones('box', low=[-3, -3], up=[3, 3]),
                    Zones('box', low=[-2, -2], up=[2, 2]),
                    Zones('box', low=[-4, -4], up=[4, 4])
                    ]
    I_zones_list = [Zones('box', low=[random.randint(-2,2), random.randint(-2,2)], up=[random.randint(-2,2), random.randint(-2,2)]),
                    Zones('box', low=[random.randint(0,2), random.randint(0,2)], up=[random.randint(0,2), random.randint(0,2)]),
                    Zones('box', low=[random.randint(-3,3), random.randint(-3,3)], up=[random.randint(-3,3), random.randint(-3,3)]),
                    Zones('box', low=[random.randint(-2,2), random.randint(-2,2)], up=[random.randint(-2,2), random.randint(-2,2)]),
                    Zones('ball', center=[random.randint(-4,4),random.randint(-4,4)], r=random.randint(0,4))
                    ]
    U_zones_list = [Zones('box', low=[random.randint(-2,2), random.randint(-2,2)], up=[random.randint(-2,2), random.randint(-2,2)]),
                    Zones('box', low=[random.randint(0,2), random.randint(0,2)], up=[random.randint(0,2), random.randint(0,2)]),
                    Zones('ball', center=[random.randint(-2,2), random.randint(-2,2)], r=random.randint(0,2)),
                    Zones('box', low=[random.randint(-2,2), random.randint(-2,2)], up=[random.randint(-2,2), random.randint(-2,2)]),
                    Zones('ball', center=[random.randint(-4,4),random.randint(-4,4)], r=random.randint(0,4))
                    ]
    f_list = [[lambda x, u: x[1],
               lambda x, u: -x[0] + u
               ],
              [lambda x, u: u - 0.5 * x[0] ** 3,
               lambda x, u: 3 * x[0] - x[1],
               ],
              [lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u,
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7),
               ],
              [lambda x, u: -x[0] + x[0] * x[1],
               lambda x, u: u - x[0] + 0.25 * x[1],
               ],
              [lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               ]
              ]
    path_list = ['dim2_1/model', 'dim2_2/model', 'dim2_3/model','dim2_4/model','dim2_5/model']
    for key in range(5, 50):
        feed = random.randint(0, 2)
        examples.update({
            key: Example(
                n_obs=2,
                D_zones=D_zones_list[feed],
                I_zones=I_zones_list[feed],
                U_zones=U_zones_list[feed],
                f=f_list[feed],
                B=None,
                u=3,
                degree=2,
                path=path_list[feed],
                dense=5,
                units=30,
                activation='relu',
                id=key,
                k=50,
            )

        }
        )
    return Env(examples[i])


def ex_4_dim(i):
    # 实际就三个例子
    examples = {
        0: Example(  ## 当前例子为展示用例---------\cite{jarvis2003lyapunov}
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=25),
            I_zones=Zones('ball', center=[0, 0, 0, 0], r=0.25),
            U_zones=Zones('ball', center=[10, 10, 10, 10], r=5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='dim4_0_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=50,
        ),

        1: Example(  # ----------\cite{Chesi04}
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
            U_zones=Zones('ball', center=[-2, -2, -2, -2], r=2),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='dim4_1_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            k=100,
        )
        # ,
        # 2: Example(  # 随机参数1 成功， 随机参数2 失败-----\cite{jin2020neural}
        #     n_obs=4,
        #     # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
        #     # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
        #     # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),
        #
        #     D_zones=Zones('box', low=[-1.3] * 4, up=[1.3] * 4),
        #     I_zones=Zones('box', low=[-0.8] * 4, up=[0.8] * 4),
        #     U_zones=Zones('box', low=[0.9] * 4, up=[1.3] * 4),
        #     f=[lambda x, u: x[2],
        #        lambda x, u: x[3],
        #        lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
        #        lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
        #                    (1 + sin(x[1])) ** 2)],
        #     B=None,  ## 没有障碍函数写 None
        #     u=0.2,
        #     degree=3,
        #     path='Cartpole/model',
        #     dense=5,
        #     units=30,
        #     activation='relu',
        #     id=1,
        #     k=50
        # )
    }

    a = random.random()
    # b = random.random()*2
    # D_zones_list = [Zones('ball', center=[0, 0, 0, 0], r=25), Zones('ball', center=[0, 0, 0, 0], r=16),
    #                 Zones('box', low=[-1.3] * 4, up=[1.3] * 4)]
    D_zones_list = [Zones('ball', center=[0, 0, 0, 0], r=25), Zones('ball', center=[0, 0, 0, 0], r=16)]
    # I_zones_list = [Zones('ball', center=[0, 0, 0, 0], r=(0.5 + a) * 2),
    #                 Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2 + a, 0.2 + a, 0.2 + a, 0.2 + a]),
    #                 Zones('box', low=[-0.8] * 4, up=[0.8 + a] * 4)]
    I_zones_list = [Zones('ball', center=[0, 0, 0, 0], r=random.uniform(1,25)),
                    Zones('box', low=[random.uniform(-4,0)]*4, up=[random.uniform(0,4)]*4)]
    # U_zones_list = [Zones('ball', center=[10, 10, 10, 10], r=5 + a), Zones('ball', center=[-2, -2, -2, -2], r=2 + a),
    #                 Zones('box', low=[0.9] * 4, up=[1.3 + a] * 4)]
    U_zones_list = [Zones('ball', center=[10, 10, 10, 10], r=random.uniform(1,15)), Zones('ball', center=[-2, -2, -2, -2], r=random.uniform(1,14))]
    f_list = [[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
              [lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
              # [lambda x, u: x[2],
              #  lambda x, u: x[3],
              #  lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
              #  lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
              #              (1 + sin(x[1])) ** 2)]
              ]
    path_list = ['dim4_test_0/model', 'dim4_test_1/model']
    for key in range(2, 50):
        feed = random.randint(0, 1)
        examples.update({
            key: Example(
                n_obs=4,
                D_zones=D_zones_list[feed],
                I_zones=I_zones_list[feed],
                U_zones=U_zones_list[feed],
                f=f_list[feed],
                B=None,
                u=3,
                degree=2,
                path=path_list[feed],
                dense=5,
                units=30,
                activation='relu',
                id=17,
                k=100,
            )

        }
        )

    return Env(examples[i])


def ex_6_dim(i):
    # 实际就两个例子
    examples = {
        0: Example(  ## 当前例子为展示用例---\cite{bouissou2014computation}
            n_obs=6,
            D_zones=Zones('box', low=[0, 0, 0, 0, 0, 0], up=[10, 10, 10, 10, 10, 10]),
            I_zones=Zones('box', low=[3, 3, 3, 3, 3, 3], up=[3.1, 3.1, 3.1, 3.1, 3.1, 3.1]),
            U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_0/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=8,
            k=100,
        ),
        1: Example(  # ---\cite{setta20}
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[1] * 6, up=[2] * 6),
            U_zones=Zones('box', low=[-1] * 6, up=[-0.5] * 6),
            f=[lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=100,
        ),
        2: Example(  # ---\cite{zhikunshe2010}
            n_obs=6,
            D_zones=Zones('box', low=[-0.8] * 6, up=[0.8] * 6),
            I_zones=Zones('box', low=[0] * 6, up=[0.2] * 6),
            U_zones=Zones('box', low=[-0.7] * 6, up=[0.7] * 6),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 - 6 * x[2] * x[3] + u,
               lambda x, u: -x[0] -x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[3] * x[5],
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=100,
        ),
        3: Example(  # ---\cite{djaballah2017construction}
            n_obs=6,
            D_zones=Zones('box', low=[0,0,2,0,0,0] , up=[10] * 6),
            I_zones=Zones('ball', center=[0, 3.05,3.05,3.05,3.05,3.05], r=0.01),
            U_zones=Zones('ball', center=[0, 7.05,7.05,7.05,7.05,7.05], r=0.01),
            f=[lambda x, u: -x[0]+4*x[1]-6*x[2]*x[3]+ u,
               lambda x, u: -x[0]-x[1]+x[4]**3,
               lambda x, u: x[0]*x[3]-x[2]+x[3]*x[5],
               lambda x, u: x[0]*x[2]+x[2]*x[5]-x[3]**3,
               lambda x, u: -2*x[1]**3-x[4]+x[5],
               lambda x, u: -3*x[2]*x[3]-x[4]**3-x[5]
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=100,
        ),
        4: Example(  # ---\cite{huang2017probabilistic}
            n_obs=6,
            D_zones=Zones('box', low=[-4.5]*6, up=[4.5] * 6),
            I_zones=Zones('box', low=[-4,-0.5,-2.4,-2.5,0,-4], up=[4,4.5,2.4,3.5,2,0]),
            U_zones=Zones('ball', center=[0, 3,3,3,3,3], r=0.5),
            f=[lambda x, u: x[3] + u,
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: (-2/3)*x[0]+(1/3)*x[0]*x[2],
               lambda x, u: (1/3)*x[0]+(2/3)*x[1]+(1/3)*x[1]*x[2],
               lambda x, u: -4+x[2]**2+8*x[4]
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_4/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=100,
        )
    }

    a = random.random()
    b = random.random() * 2
    D_zones_list = [Zones('box', low=[0, 0, 0, 0, 0, 0], up=[10, 10, 10, 10, 10, 10]),
                    Zones('box', low=[-2] * 6, up=[2] * 6),
                    Zones('box', low=[-0.8] * 6, up=[0.8] * 6),
                    Zones('box', low=[0,0,2,0,0,0] , up=[10] * 6),
                    Zones('box', low=[-4.5]*6, up=[4.5] * 6)]
    # I_zones_list = [Zones('box', low=[random.uniform(0, 4)]*6, up=[ random.uniform(0, 4)]*6),
    #                 Zones('box', low=[random.uniform(-2, 0)] * 6, up=[random.uniform(-2, 0)] * 6),
    #                 Zones('box', low=[random.uniform(-0.8, 0)] * 6, up=[random.uniform(-0.8, 0)] * 6),
    #                 Zones('ball', center=[0, random.uniform(7.04,8),random.uniform(7.04,8),random.uniform(7.04,8),random.uniform(7.04,8),random.uniform(7.04,8)], r=0.01),
    #                 Zones('box', low=[random.uniform(-4.5,3.5)]*6, up=[random.uniform(2.5,4.5)]*6)
    #                 ]

    I_zones_list = [Zones('box', low=[random.uniform(0, 10)] * 6, up=[random.uniform(0, 10)] * 6),
                    Zones('box', low=[random.uniform(-2, 2)] * 6, up=[random.uniform(-2, 2)] * 6),
                    Zones('box', low=[random.uniform(-0.8, 0.8)] * 6, up=[random.uniform(-0.8, 0.8)] * 6),
                    Zones('ball', center=[0, random.uniform(0, 10), random.uniform(2, 10), random.uniform(0, 10),
                                          random.uniform(0, 10), random.uniform(0, 10)], r=0.01+a),
                    Zones('box', low=[random.uniform(-4.5, 4.5)] * 6, up=[random.uniform(-4.5, 4.5)] * 6)
                    ]

    # U_zones_list = [
    #     Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5], up=[4.1 + random.uniform(0, 5), 4.2 + random.uniform(0, 5), 4.3 + random.uniform(0, 5), 4.4 + random.uniform(0, 5), 4.5 + random.uniform(0, 5), 4.6 + random.uniform(0, 5)]),
    #     Zones('box', low=[random.uniform(0.1, 2)] * 6, up=[random.uniform(0.1, 2)] * 6),
    #     Zones('box', low=[random.uniform(0.1, 0.8)] * 6, up=[random.uniform(0.1, 0.8)] * 6),
    #     Zones('ball', center=[0, random.uniform(8.05,10),random.uniform(8.05,10),random.uniform(8.05,10),random.uniform(8.05,10),random.uniform(8.05,10)], r=0.01),
    #     Zones('ball', center=[0, random.uniform(0,3),random.uniform(0,3),random.uniform(0,3),random.uniform(0,3),random.uniform(0,3)], r=0.5)
    #     ]

    U_zones_list = [
        Zones('box', low=[random.uniform(0, 10)]*6,
              up=[random.uniform(0, 10)]*6),
        Zones('box', low=[random.uniform(-2, 2)] * 6, up=[random.uniform(-2, 2)] * 6),
        Zones('box', low=[random.uniform(-0.8, 0.8)] * 6, up=[random.uniform(-0.8, 0.8)] * 6),
        Zones('ball', center=[0, random.uniform(0, 10), random.uniform(2, 10), random.uniform(0, 10),
                                          random.uniform(0, 10), random.uniform(0, 10)], r=0.01+a),
        Zones('ball', center=[0, random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5),
                              random.uniform(-4.5, 4.5)], r=0.5+a)
    ]

    f_list = [[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
               ],
              [lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u
               ],
              [lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 - 6 * x[2] * x[3] + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[3] * x[5],
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
               ],
              [lambda x, u: -x[0] + 4 * x[1] - 6 * x[2] * x[3] + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[3] * x[5],
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5]
               ],
              [lambda x, u: x[3] + u,
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: (-2 / 3) * x[0] + (1 / 3) * x[0] * x[2],
               lambda x, u: (1 / 3) * x[0] + (2 / 3) * x[1] + (1 / 3) * x[1] * x[2],
               lambda x, u: -4 + x[2] ** 2 + 8 * x[4]
               ]
              ]
    path_list = ['dim6_test_0/model', 'dim6_test_1/model','dim6_test_2/model','dim6_test_3/model','dim6_test_4/model']
    for key in range(5, 50):
        feed = random.randint(0, 4)
        examples.update({
            key: Example(
                n_obs=6,
                D_zones=D_zones_list[feed],
                I_zones=I_zones_list[feed],
                U_zones=U_zones_list[feed],
                f=f_list[feed],
                B=None,
                u=3,
                degree=2,
                path=path_list[feed],
                dense=5,
                units=30,
                activation='relu',
                id=key,
                k=100,
            )

        }
        )

    return Env(examples[i])



def ex_9_dim(i):
    # 实际就1个例子
    examples = {
        0: Example(  # ----原本例子-----#----\cite{chen2020novelchen2020novel}
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
            id=0,
            k=100  # 3000
        )
    }
    for key in range(1, 50):
        random_a = 0 + (0.99 - 0) * np.random.random()
        random_b = np.random.random()
        examples.update({
            key: Example(  # ----\cite{chen2020novelchen2020novel}
                n_obs=9,
                D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
                # I_zones=Zones('box', low=[0.99 - random_a] * 9, up=[1.01 - random_a] * 9),
                # U_zones=Zones('box', low=[1.8 + random_b] * 9, up=[2 + random_b] * 9),
                # I_zones=Zones('box', low=[-2 +random_a +0.0000001] * 9, up=[1.01] * 9),
                # I_zones=Zones('box', low=[random.uniform(-2,0.99)] * 9, up=[random.uniform(1.01,1.8-0.01)] * 9),
                # U_zones=Zones('box', low=[1.8] * 9, up=[2] * 9),
                I_zones=Zones('box', low=[random.uniform(-2,2)] * 9, up=[random.uniform(-2,2)] * 9),
                U_zones=Zones('box', low=[random.uniform(-2,2)] * 9, up=[random.uniform(-2,2)] * 9),

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
                id=key,
                k=100  # 3000
            )
        })

    return Env(examples[i])


def ex_12_dim(i):
    # 实际就2个例子
    examples = {
        0: Example(  # 在train 0的基础上改变不安全区域---\cite{chen2020novel}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.6, 0.6, 0.6, 1.6, 1.6, 1.6, 1.6, -0.5, 1.5, 1.5, -0.1, 2.0]),
            f=[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
               lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
               lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
               lambda x, u: x[9],
               lambda x, u: x[10],
               lambda x, u: x[11],
               lambda x, u: 9.81 * x[1],
               lambda x, u: -9.81 * x[0],
               lambda x, u: -16.3541 * x[11] + u
               ],
            B=None,
            u=3,
            degree=3,
            path='dim12test_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=16,
            k=50  # 6000
        ),
        1: Example(  # 在0的基础上改变初始区域-----\cite{setta20}
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[1] * 12, up=[2] * 12),
            U_zones=Zones('box', low=[-1] * 12, up=[-0.5] * 12),
            f=[lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=2,
            path='dim12_test_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=17,
            k=100,
        ),
    }
    a = random.random()
    b = random.random() * 0.1
    # I_zones_list = [Zones('box', low=[-0.1 + b] * 12, up=[0.1] * 12), Zones('box', low=[1 - a] * 12, up=[2] * 12)]
    # U_zones_list = [Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
    #                       up=[0.6 + a, 0.6 + a, 0.6 + a, 1.6 + a, 1.6 + a, 1.6 + a, 1.6 + a, -0.5 - a, 1.5 + a, 1.5 + a,
    #                           -0.1 + a, 2.0 + a]),
    #                 Zones('box', low=[-1] * 12, up=[-0.5 + a] * 12)]
    I_zones_list = [Zones('box', low=[random.uniform(-2,2)] * 12, up=[random.uniform(-2,2)] * 12), Zones('box', low=[random.uniform(-2,2)] * 12, up=[random.uniform(-2,2)] * 12)]
    U_zones_list = [Zones('box', low=[random.uniform(-2,2)] * 12, up=[random.uniform(-2,2)] * 12), Zones('box', low=[random.uniform(-2,2)] * 12, up=[random.uniform(-2,2)] * 12)]

    f_list = [[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: -7253.4927 * x[0] + 1936.3639 * x[10] - 1338.7624 * x[3] + 1333.3333 * x[7],
               lambda x, u: -1936.3639 * x[9] - 7253.4927 * x[1] - 1338.7624 * x[4] - 1333.3333 * x[6],
               lambda x, u: -769.2308 * x[2] - 770.2301 * x[5],
               lambda x, u: x[9],
               lambda x, u: x[10],
               lambda x, u: x[11],
               lambda x, u: 9.81 * x[1],
               lambda x, u: -9.81 * x[0],
               lambda x, u: -16.3541 * x[11] + u
               ],
              [lambda x, u: x[0] * x[2],
               lambda x, u: x[0] * x[4],
               lambda x, u: (x[3] - x[2]) * x[2] - 2 * x[4] ** 2,
               lambda x, u: -(x[3] - x[2]) ** 2 + (-x[0] ** 2 + x[5] ** 2),
               lambda x, u: x[1] * x[5] + (x[2] - x[3]) * x[4],
               lambda x, u: 2 * x[1] * x[4] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
              ]
    path_list = ['dim12test_0/model', 'dim12_test_1/model']
    for key in range(2, 50):
        feed = random.randint(0, 1)
        examples.update({
            key: Example(
                n_obs=12,
                D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
                I_zones=I_zones_list[feed],
                U_zones=U_zones_list[feed],
                f=f_list[feed],
                B=None,
                u=3,
                degree=2,
                path=path_list[feed],
                dense=5,
                units=30,
                activation='relu',
                id=17,
                k=100,
            )

        }
        )

    return Env(examples[i])
