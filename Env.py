import numpy as np
from sympy import *
import math
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

# 6维以下统一为6维
def uni_6dim_train(i):
    examples = {
        0: Example(  ## 当前例子为2维转6维
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[1.3, -0.1,1.3, -0.1,1.3, -0.1], up=[1.35, 0,1.35, 0,1.35, 0]),
            U_zones=Zones('box', low=[-2, -2,-2, -2,-2, -2], up=[-1.9, -1.9,2,2,2,2]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],

            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='uni6dim_train_0/model',
            dense=4,
            units=20,
            activation='relu',
            id=2,
            k=50
        ),
        1: Example( #3维转6维
            n_obs=6,
            D_zones=Zones(shape='box', low=[-2.2] * 6, up=[2.2] * 6),
            I_zones=Zones(shape='box', low=[-0.4]* 6, up=[0.4] * 6),
            U_zones=Zones(shape='box', low=[2, 2, 2,-2.2,-2.2,-2.2], up=[2.2] * 6),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  # lambda x: 306.5783213 + 35.3288 * x[0] - 122.5043 * x[1] + 217.9696 * x[2] - 16.8297 * x[
            #     0] ** 2 + 11.0428 * x[0] * x[1] + 39.0244 * x[0] * x[2] - 169.7252 * x[1] ** 2 - 185.8183 * x[1] * x[
            #                 2] - 29.7622 * x[2] ** 2,  ## 没有障碍函数写 None
            u=3,
            degree=3,
            path='uni6dim_train_1/model',
            dense=4,
            units=20,
            activation='relu',
            id=0,
            k=50,
        ),
        2: Example( # 6 dim
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
            path='uni6dim_train_2/model',
            dense=4,
            units=20,
            activation='sigmoid',
            id=7,
            k=100,
        ),
        3: Example( # 2dim to 6 dim
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[-0.1] * 6, up=[0] * 6),
            U_zones=Zones('box', low=[1.2, -0.1,-2,-2,-2,-2], up=[1.3, 0.1,2, 2,2, 2]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='uni6dim_train_3/model',
            dense=4,
            units=20,
            activation='relu',
            id=3,
            k=50,
        )

    }
    return Env(examples[i])

def uni_6dim_test(i):
    examples = {
        0: Example(  ## 当前例子为2维转6维
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[-0.1] * 6, up=[0] * 6),
            U_zones=Zones('box', low=[1.2, -0.1, -2, -2, -2, -2], up=[1.3, 0.1, 2, 2, 2, 2]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='uni6dim_test_0/model',
            dense=4,
            units=20,
            activation='relu',
            id=3,
            k=50,
        ),
        1: Example( #4维转6维
            n_obs=6,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('ball', center=[0, 0, 0, 0, 0, 0], r=25),
            I_zones=Zones('ball', center=[0, 0, 0, 0,0,0], r=0.25),
            U_zones=Zones('box', low=[5, 5,5, 5, -25 , -25], up=[15,15,15,15, 25,25]),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2]),
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='uni6dim_test_1/model',
            dense=4,
            units=20,
            activation='relu',
            id=1,
            k=50,
        ),
        2: Example( # 6 dim
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
            path='uni6dim_test_2/model',
            dense=4,
            units=20,
            activation='sigmoid',
            id=7,
            k=100,
        ),
        3: Example( # nonlinear 2dim to 6 dim
            n_obs=6,
            D_zones=Zones('box', low=[-3.15, -5, -5, -5, -5, -5], up=[3.15, 5, 5, 5, 5, 5]),
            I_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            U_zones=Zones('box', low=[2.5, 2.5,-5,-5,-5,-5], up=[3, 3,5,5,5,5]),
            f=[lambda x, u: x[1],
               lambda x, u: - 10 * sin(x[0]) - 0.1 * x[1] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='uni6dim_test_3/model',
            dense=4,
            units=20,
            activation='relu',
            id=2,
            k=50
        )

    }
    return Env(examples[i])

def uni_6dimto12_train(i):
    examples = {
        0: Example(  ## 当前例子为2维转6维
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[1.3, -0.1, 1.3, -0.1, 1.3, -0.1,0,0,0,0,0,0], up=[1.35, 0, 1.35, 0, 1.35, 0,0,0,0,0,0,0]),
            U_zones=Zones('box', low=[-2]*12, up=[-1.9, -1.9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],

            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='uni6dimto12_train_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        1: Example(  # 3维转6维
            n_obs=12,
            D_zones=Zones(shape='box', low=[-2.2] * 12, up=[2.2] * 12),
            I_zones=Zones(shape='box', low=[-0.4] * 12, up=[0.4] * 12),
            U_zones=Zones(shape='box', low=[2, 2, 2, -2.2, -2.2, -2.2, -2.2, -2.2, -2.2, -2.2, -2.2, -2.2], up=[2.2] * 12),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  # lambda x: 306.5783213 + 35.3288 * x[0] - 122.5043 * x[1] + 217.9696 * x[2] - 16.8297 * x[
            #     0] ** 2 + 11.0428 * x[0] * x[1] + 39.0244 * x[0] * x[2] - 169.7252 * x[1] ** 2 - 185.8183 * x[1] * x[
            #                 2] - 29.7622 * x[2] ** 2,  ## 没有障碍函数写 None
            u=3,
            degree=3,
            path='uni6dimto12_train_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=0,
            k=50,
        ),
        2: Example(  # 6 dim
            n_obs=12,
            D_zones=Zones('box', low=[0] * 12, up=[10] * 12),
            I_zones=Zones('box', low=[3,3,3,3,3,3,5,5,5,5,5,5], up=[3.1,3.1,3.1,3.1,3.1,3.1,5,5,5,5,5,5]),
            U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5,0,0,0,0,0,0], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6,10,10,10,10,10,10]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni6dimto12_train_2/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=100,
        ),
        3: Example(  # 2dim to 6 dim
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0] * 12),
            U_zones=Zones('box', low=[1.2, -0.1, -0.1,-0.1, -0.1,-0.1, -0.1,-0.1, -0.1,-0.1, -0.1,-0.1,], up=[1.3, 0.1,0,0,0,0,0,0,0,0,0,0]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='uni6dimto12_train_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=3,
            k=50,
        )

    }
    return Env(examples[i])

def metarl_6dim_train(i):
    examples = {
        0: Example(  ## 当前例子为2维转6维
            n_obs=2,
            D_zones=Zones('box', low=[-2] * 2, up=[2] * 2),
            I_zones=Zones('box', low=[1.3, -0.1], up=[1.35, 0]),
            U_zones=Zones('box', low=[-2, -2], up=[-1.9, -1.9]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u
               ],

            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='matrl6_train_0/model',
            dense=4,
            units=20,
            activation='relu',
            id=2,
            k=50
        ),
        1: Example( #3维转6维
            n_obs=4,
            D_zones=Zones(shape='box', low=[-2.2] * 4, up=[2.2] * 4),
            I_zones=Zones(shape='box', low=[-0.4]* 4, up=[0.4] * 4),
            U_zones=Zones(shape='box', low=[2] * 4, up=[2.2] * 4),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u,
               lambda x, u: 0
               ],
            B=None,  # lambda x: 306.5783213 + 35.3288 * x[0] - 122.5043 * x[1] + 217.9696 * x[2] - 16.8297 * x[
            #     0] ** 2 + 11.0428 * x[0] * x[1] + 39.0244 * x[0] * x[2] - 169.7252 * x[1] ** 2 - 185.8183 * x[1] * x[
            #                 2] - 29.7622 * x[2] ** 2,  ## 没有障碍函数写 None
            u=3,
            degree=3,
            path='metarl6_train_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=0,
            k=50,
        ),
        2: Example( # 6 dim
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
            path='metarl6_train_2/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=50,
        ),
        3: Example( # 2dim to 6 dim
            n_obs=2,
            D_zones=Zones('box', low=[-2] * 2, up=[2] * 2),
            I_zones=Zones('box', low=[-0.1] * 2, up=[0] * 2),
            U_zones=Zones('box', low=[1.2, -0.1], up=[1.3, 0.1]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='metarl6_train_3/model',
            dense=4,
            units=20,
            activation='relu',
            id=3,
            k=50,
        )

    }
    return Env(examples[i])

# 12维以下统一为12维
def uni_12dim_train(i):
    examples = {
        0: Example( # 12 dim
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, -0.5, 1.5, 1.5, -0.5, 1.5]),
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
            path='uni12dim_train_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=50  # 6000
        ),
        1: Example( #8维转12维
            n_obs=12,
            D_zones=Zones('ball', center=[0] * 12, r=4),
            I_zones=Zones('ball', center=[1] * 12, r=0.25),
            U_zones=Zones('box', low=[-2.16,-2.16,-2.16,-2.16,-2.16,-2.16,-2.16,-2.16,-4,-4,-4,-4],
                                  up=[-1.84,-1.84,-1.84,-1.84,-1.84,-1.84,-1.84,-1.84,4,4,4,4]),
            f=[lambda x, u: - 567.0 / 2400.0,
               lambda x, u: (- 2400 * x[0] - 567 ) / 4180,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1]) / 3980,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2]),
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3]) / 800,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3] - 800 * x[4]),
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3] - 800 * x[4] - 170 * x[5]) / 20 ,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3] - 800 * x[4] - 170 * x[5] - 20 * x[6]),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni12dim_train_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        2: Example( # 9 dim to 12 dim
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.66] * 12, up=[0.90] * 12),
            U_zones=Zones('box', low=[1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,-2,-2,-2], up=[2] * 12),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni12dim_train_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        3: Example( # 7dim to 12dim
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.99] * 12,up=[1.01] * 12),
            U_zones=Zones('box', low=[1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8,-2,-2,-2,-2,-2], up=[2] * 12),
            f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
               lambda x, u: 0.4 * x[0] - x[1],
               lambda x, u: x[1] - 5 * x[2] * x[3],
               lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
               lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
               lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
               lambda x, u: -0.5 * x[6] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni12dim_train_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=8,
            k=100
        )

    }
    return Env(examples[i])


def uni_12dim_test(i):
    examples = {
        0: Example( # 12 dim change unsafe zone
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[-1, -1, -1, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, -0.5, 1.5, 1.5, -0.5, 1.5]),
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
            path='uni12dim_train_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=50  # 6000
        ),
        1: Example( #8维转12维 change init zone
            n_obs=12,
            D_zones=Zones('ball', center=[0] * 12, r=4),
            I_zones=Zones('ball', center=[1] * 12, r=0.1),
            U_zones=Zones('box', low=[-2.16,-2.16,-2.16,-2.16,-2.16,-2.16,-2.16,-2.16,-4,-4,-4,-4],
                                  up=[-1.84,-1.84,-1.84,-1.84,-1.84,-1.84,-1.84,-1.84,4,4,4,4]),
            f=[lambda x, u: - 567.0 / 2400.0,
               lambda x, u: (- 2400 * x[0] - 567 ) / 4180,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1]) / 3980,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2]),
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3]) / 800,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3] - 800 * x[4]),
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3] - 800 * x[4] - 170 * x[5]) / 20 ,
               lambda x, u: (- 2400 * x[0] - 567 -4180 * x[1] - 3980 * x[2] - x[3] - 800 * x[4] - 170 * x[5] - 20 * x[6]),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni12dim_train_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        2: Example( # 9 dim to 12 dim change unsafe zone
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.99] * 12, up=[1.1] * 12),
            U_zones=Zones('box', low=[1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,-2,-2,-2], up=[2] * 12),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni12dim_train_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        3: Example( # 7dim to 12dim  change unsafe zone
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.99] * 12,up=[1.01] * 12),
            U_zones=Zones('box', low=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,-2,-2,-2,-2,-2], up=[2] * 12),
            f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
               lambda x, u: 0.4 * x[0] - x[1],
               lambda x, u: x[1] - 5 * x[2] * x[3],
               lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
               lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
               lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
               lambda x, u: -0.5 * x[6] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='uni12dim_train_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=8,
            k=800
        )

    }
    return Env(examples[i])

def get_2dimTo4dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
            n_obs=4,
            D_zones=Zones('box', low=[-2] * 4, up=[2] * 4),
            I_zones=Zones('box', low=[1.3, -0.1,0,0], up=[1.35, 0, 0, 0]),
            U_zones=Zones('box', low=[-2, -2,-2,-2], up=[-1.9, -1.9,2,2]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='text_2to4_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        1: Example(
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=4),
            I_zones=Zones('box', low=[1, -0.2,1,1], up=[1.4, 0.2,1.4,1.4]),
            U_zones=Zones('box', low=[-2, -0.2,-4,-4], up=[-1.5, 0.2,4,4]),
            f=[lambda x, u: u - 0.5 * x[0] ** 3,
               lambda x, u: 3 * x[0] - x[1],
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='test_2to4_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            k=50,
        ),
        2: Example(
            n_obs=2,
            D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
            I_zones=Zones('box', low=[-1, 1], up=[-0.9, 1.1]),
            U_zones=Zones('ball', center=[-2.25, -1.75], r=0.25),

            f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u,
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7),
               ],
            B=None,  ## 没有障碍函数写 None
            u=0.3,
            degree=2,
            path='testExp2/model',
            dense=4,
            units=20,
            activation='relu',
            id=6,
            k=50
        ),
        3: Example(
            n_obs=2,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[-1.5, -1.5], up=[-1.4, -1.3]),
            U_zones=Zones('box', low=[-0.1, 0.5], up=[0.1, 1]),
            f=[lambda x, u: -x[0] + x[0] * x[1],
               lambda x, u: u - x[0] + 0.25 * x[1],
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='testExp3/model',
            dense=4,
            units=20,
            activation='relu',
            id=3,
            k=50,
        )

    }
    return Env(examples[i])
# 各种困难例子的测试集 3个
def get_singleTest(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
            n_obs=2,
            D_zones=Zones('box', low=[-3.15, -5], up=[3.15, 5]),
            I_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            U_zones=Zones('box', low=[2.5, 2.5], up=[3, 3]),
            f=[lambda x, u: x[1],
               lambda x, u: - 10 * sin(x[0]) - 0.1 * x[1] + u],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='textNonlinear0/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        1: Example(
            n_obs=6,
            D_zones=Zones('box', low=[-1] * 6, up=[1.0] * 6),
            I_zones=Zones('box', low=[-0.5] * 6, up=[0.5] * 6),
            U_zones=Zones('box', low=[0.9] * 6, up=[1.0] * 6),
            f=[lambda x, u: x[3],
               lambda x, u: x[4],
               lambda x, u: x[5],
               lambda x, u: - (u[0] + u[1]) * sin(x[2]),
               lambda x, u: 10 * (u[0] + u[1]) * cos(x[2]) - 0.1,
               lambda x, u: u[0] - u[1]
               ],
            # x0 = x  x1 = y x2 = xita, x3 = x' x4 = y' x5 = xita'
            B=None,
            u=3,
            degree=3,
            path='UAV/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=400,
        ),
        2: Example( #随机参数1 成功， 随机参数2 失败
            n_obs=4,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('box', low=[-1.3] * 4, up=[1.3] * 4),
            I_zones=Zones('box', low=[-0.8] * 4, up=[0.8] * 4),
            U_zones=Zones('box', low=[0.9] * 4, up=[1.3] * 4),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
               lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / ((1 + sin(x[1])) ** 2)],
            B=None,  ## 没有障碍函数写 None
            u=0.2,
            degree=3,
            path='Cartpole/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50
        ),
        3: Example(
            n_obs=2,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('box', low=[-1.3] * 2, up=[1.3] * 2),
            I_zones=Zones('box', low=[-0.8] * 2, up=[0.8] * 2),
            U_zones=Zones('box', low=[0.9] * 2, up=[1.3] * 2),
            f=[lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
               lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
                           (1 + sin(x[1])) ** 2)],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='Cartpole2_gai/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50
        ),
        4: Example( # 初始参数1 成功 #初始参数2 失败 # 参数3 前期失败，后期成功 失败例子1
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
            id=5,
            k=50,
        ),
        5: Example(  # 在上一个例子基础上改 失败例子2
            n_obs=2,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('ball', center=[1, 0], r=0.2),
            U_zones=Zones('ball', center=[-1, 1], r=3.25),
            f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='trainExp1_gai/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            k=50,
        ),
        6: Example(  # 在上一个4维失败案例基础上改:扩大初始区域，缩小不安全区域
            n_obs=4,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('box', low=[-1.3] * 4, up=[1.3] * 4),
            I_zones=Zones('box', low=[-1] * 4, up=[1] * 4),
            U_zones=Zones('box', low=[1.2] * 4, up=[1.3] * 4),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
               lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
                           (1 + sin(x[1])) ** 2)],
            B=None,  ## 没有障碍函数写 None
            u=0.2,
            degree=3,
            path='Cartpole_gai/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50
        ),
    }
    return Env(examples[i])


# 2dim 的训练集 4个
def get_trainEnv_2dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
            n_obs=2,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
            U_zones=Zones('ball', center=[3, 2], r=1),
            f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] * u,
               lambda x, u: -2 * x[1] - x[0] ** 2 + u],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='trainExp0/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        1: Example(
            n_obs=2,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('ball', center=[1, 0], r=0.25),
            U_zones=Zones('ball', center=[-1, 1], r=0.25),
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
            id=5,
            k=50,
        ),
        2: Example(
            n_obs=2,
            # D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
            # I_zones=Zones('ball', center=[1, 1], r=0.25),
            # U_zones=Zones('ball', center=[-1, -1], r=0.36),
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('ball', center=[1, 1], r=0.5),
            U_zones=Zones('ball', center=[-2, -2], r=0.5),

            f=[lambda x, u: x[1],
               lambda x, u: -0.5 * x[0] ** 2 - x[1] + u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=0.3,
            degree=3,
            path='trainExp2/model',
            dense=5,
            units=30,
            activation='relu',
            id=6,
            k=50
        ),
        3: Example(
            n_obs=2,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[-0.1, -0.1], up=[0, 0]),
            U_zones=Zones('box', low=[1.2, -0.1], up=[1.3, 0.1]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='trainExp3/model',
            dense=5,
            units=30,
            activation='relu',
            id=3,
            k=50,
        )

    }
    return Env(examples[i])

# 2dim 的测试集 4个
def get_testEnv_2dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
            n_obs=2,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[1.3, -0.1], up=[1.35, 0]),
            U_zones=Zones('box', low=[-2, -2], up=[-1.9, -1.9]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] + u],
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
        1: Example(
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
        2: Example(
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
        3: Example(
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
        )

    }
    return Env(examples[i])

def get_trainEnv_3dim(i):
    example = {
        0: Example(  ## 当前例子为展示用例
            n_obs=3,
            D_zones=Zones(shape='box', low=[-2.2, -2.2, -2.2], up=[2.2, 2.2, 2.2]),
            I_zones=Zones(shape='box', low=[-0.4, -0.4, -0.4], up=[0.4, 0.4, 0.4]),
            U_zones=Zones(shape='box', low=[2, 2, 2], up=[2.2, 2.2, 2.2]),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u,  ##--+
               ],
            B=None,  # lambda x: 306.5783213 + 35.3288 * x[0] - 122.5043 * x[1] + 217.9696 * x[2] - 16.8297 * x[
            #     0] ** 2 + 11.0428 * x[0] * x[1] + 39.0244 * x[0] * x[2] - 169.7252 * x[1] ** 2 - 185.8183 * x[1] * x[
            #                 2] - 29.7622 * x[2] ** 2,  ## 没有障碍函数写 None
            u=3,
            degree=3,
            path='dim3_0/model',
            dense=4,
            units=30,
            activation='relu',
            id=0,
            k=50,
        ),
        1: Example(  ## 当前例子为展示用例
            n_obs=3,
            D_zones=Zones(shape='box', low=[-2.2, -2.2, -2.2], up=[2.2, 2.2, 2.2]),
            I_zones=Zones(shape='box', low=[-0.4, -0.4, -0.4], up=[0.4, 0.4, 0.4]),
            U_zones=Zones(shape='box', low=[2, 2, 2], up=[2.2, 2.2, 2.2]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u,  ##--+
               ],
            B=None,  # lambda x: 306.5783213 + 35.3288 * x[0] - 122.5043 * x[1] + 217.9696 * x[2] - 16.8297 * x[
            #     0] ** 2 + 11.0428 * x[0] * x[1] + 39.0244 * x[0] * x[2] - 169.7252 * x[1] ** 2 - 185.8183 * x[1] * x[
            #                 2] - 29.7622 * x[2] ** 2,  ## 没有障碍函数写 None
            u=3,
            degree=3,
            path='dim3_0/model',
            dense=4,
            units=30,
            activation='relu',
            id=0,
            k=50,
        ),
    }
# 4dim的训练集 4个
def get_trainEnv_4dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
            n_obs=4,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('ball', center=[0, 0, 0, 0], r=25),
            I_zones=Zones('ball', center=[0, 0, 0, 0], r=0.25),
            U_zones=Zones('ball', center=[1.5, 1.5, -1.5, -1.5], r=0.25),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='dim4_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50,
        ),
        1: Example(
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
            U_zones=Zones('ball', center=[-2, -2, -2, -2], r=1),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='dim4_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=100,
        ),
        2: Example(
            # 在上一个例子基础上扩大了不安全区域和初始区域
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('box', low=[-0.3, -0.3, -0.3, -0.3], up=[0.3, 0.3, 0.3, 0.3]),
            U_zones=Zones('ball', center=[-2.5, -2.5, -2.5, -2.5], r=1),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='dim4_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=100,
        ),
        3: Example(
            # 在第一个例子基础上扩大了不安全区域和初始区域，缩小不变式区间
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('ball', center=[0, 0, 0, 0], r=0.5),
            U_zones=Zones('ball', center=[1.5, 1.5, -1.5, -1.5], r=0.5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='dim4_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50,
        ),
    }
    return Env(examples[i])

# 在原训练集基础上，扩大了不安全区域
def get_testEnv_4dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
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
        1: Example(
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
        ),
        2: Example(
            # 在上一个例子基础上扩大了不安全区域和初始区域
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('box', low=[-0.3, -0.3, -0.3, -0.3], up=[0.3, 0.3, 0.3, 0.3]),
            U_zones=Zones('ball', center=[-2.5, -2.5, -2.5, -2.5], r=2),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='dim4_2_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=6,
            k=100,
        ),
        3: Example(
            # 在第一个例子基础上扩大了不安全区域和初始区域，缩小不变式区间
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('ball', center=[0, 0, 0, 0], r=0.5),
            U_zones=Zones('ball', center=[1.5, 1.5, -1.5, -1.5], r=2.5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='dim4_3_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=7,
            k=50,
        ),
    }
    return Env(examples[i])

# 6dim的训练集 4个
def get_trainEnv_6dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
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
            path='dim6_0/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=100,
        ),
        1: Example(
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
            path='dim6_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            k=100,
        ),
        2: Example(  # 在 0 的基础上改变不安全和初始区域
            n_obs=6,
            D_zones=Zones('box', low=[0] * 6, up=[10.0] * 6),
            I_zones=Zones('box', low=[2.0] * 6, up=[2.1] * 6),
            U_zones=Zones('box', low=[3, 3.1, 3.2, 3.3, 3.4, 3.5], up=[3.1, 3.2, 3.3, 3.4, 3.5, 3.6]),
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
            path='dim6_0/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=100,
        ),
        3: Example( # 在1的基础上改变初始区域
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[0] * 6, up=[1] * 6),
            U_zones=Zones('box', low=[-0.5] * 6, up=[-0.5] * 6),
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
            path='dim6_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            k=100,
        ),
    }
    return Env(examples[i])

def get_testEnv_6dim(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
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
        1: Example(
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
        2: Example(  # 在 0 的基础上改变不安全和初始区域
            n_obs=6,
            D_zones=Zones('box', low=[0] * 6, up=[10.0] * 6),
            I_zones=Zones('box', low=[2.0] * 6, up=[2.1] * 6),
            U_zones=Zones('box', low=[3, 3.1, 3.2, 3.3, 3.4, 3.5], up=[3.1, 3.2, 3.3, 3.4, 3.5, 3.6]),
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
            path='dim6_test_2/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=10,
            k=100,
        ),
        3: Example( # 在1的基础上改变初始区域
            n_obs=6,
            D_zones=Zones('box', low=[-2] * 6, up=[2] * 6),
            I_zones=Zones('box', low=[0] * 6, up=[1] * 6),
            U_zones=Zones('box', low=[-0.5] * 6, up=[0.5] * 6),
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
            path='dim6_test_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100,
        ),
    }
    return Env(examples[i])

def get_trainEnv_9dim(i):
    examples = {
        0: Example(
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
            degree=3,
            path='dim9_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        1: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.50] * 9, up=[0.90] * 9),
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
            degree=3,
            path='dim9_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        2: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.66] * 9, up=[0.90] * 9),
            U_zones=Zones('box', low=[1.9] * 9, up=[2] * 9),
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
            degree=3,
            path='dim9_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        3: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.96] * 9, up=[1.2] * 9),
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
            degree=3,
            path='dim9_3/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
        3: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.50] * 9, up=[0.90] * 9),
            U_zones=Zones('box', low=[1.6] * 9, up=[1.9] * 9),
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
            degree=3,
            path='dim9_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100  # 3000
        ),
    }
    return Env(examples[i])

def get_testEnv_9dim(i):
    examples = {
        0: Example(
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
        ),
        1: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.50] * 9, up=[0.90] * 9),
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
            path='dim9_1_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=13,
            k=100  # 3000
        ),
        2: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.66] * 9, up=[0.90] * 9),
            U_zones=Zones('box', low=[1.88] * 9, up=[2] * 9),
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
            degree=3,
            path='dim9_2_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=14,
            k=100  # 3000
        ),
        3: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.96] * 9, up=[1.2] * 9),
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
            path='dim9_3_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=15,
            k=100  # 3000
        )
    }
    return Env(examples[i])

def get_trainEnv_12dim(i):
    examples = {
        0: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, -0.5, 1.5, 1.5, -0.5, 1.5]),
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
            path='ex12_0/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=50  # 6000
        ),
        1: Example( #在0的基础上改变初始区域
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.2] * 12, up=[0.2] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, -0.5, 1.5, 1.5, -0.5, 1.5]),
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
            path='ex12_1/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=50  # 6000
        ),
        2: Example( #9dim to 12
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.50] * 12, up=[0.90] * 12),
            U_zones=Zones('box', low=[1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,-2,-2,-2], up=[1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,2,2,2]),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='exp12_2/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=50  # 3000
        ),
        3: Example(  ## 6dim0 to 12dim
            n_obs=12,
            D_zones=Zones('box', low=[0] * 12, up=[10] * 12),
            I_zones=Zones('box', low=[3] * 12, up=[3.1] * 12),
            U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5, 0, 0, 0, 0, 0, 0],
                          up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 10, 10, 10, 10, 10, 10]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6to9_test_0/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=50,
        ),
    }
    return Env(examples[i])

def get_testEnv_12dim(i):
    examples = {
        0: Example( # 在train 0的基础上改变不安全区域
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
        1: Example( #在0的基础上改变初始区域
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
    return Env(examples[i])

def get_UniTest(i):
    examples = {
        1: Example(  ## ��ǰ����Ϊչʾ����
            n_obs=12,
            D_zones=Zones('box', low=[-2]*12, up=[2]*12),
            I_zones=Zones('box', low=[1.3, -0.1,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2], up=[1.35, 0,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]),
            U_zones=Zones('box', low=[-2, -2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2], up=[-1.9, -1.9,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]),
            f=[
                lambda x, u: x[1],
                lambda x, u: -x[0] + u,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0,
                lambda x, u: 0
               ],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## û���ϰ�����д None
            B=None,
            u=1,
            degree=1,
            path='textExp0_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=0,
            k=50
        ),
        2: Example(
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=4),
            I_zones=Zones('box', low=[1, -0.2,0,0,0,0,0,0,0,0,0,0], up=[1.4, 0.2,0,0,0,0,0,0,0,0,0,0]),
            U_zones=Zones('box', low=[-2, -0.2,0,0,0,0,0,0,0,0,0,0], up=[-1.5, 0.2,0,0,0,0,0,0,0,0,0,0]),
            f=[lambda x, u: u - 0.5 * x[0] ** 3,
               lambda x, u: 3 * x[0] - x[1],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=1,
            degree=3,
            path='testExp1_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50,
        ),
        3: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-3]*12, up=[3]*12),
            I_zones=Zones('box', low=[-1, 1,-2.25,-2,-2,-2,-2,-2,-2,-2,-2,-2],
                          up=[-0.9, 1.1,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2]),
            U_zones=Zones('ball', center=[-2.25, -1.75,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2], r=0.25),

            f=[lambda x, u: -0.1 / 3 * x[0] ** 3 + 7 / 8 + u,
               lambda x, u: 0.8 * (x[0] - 0.8 * x[1] + 0.7),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=0.3,
            degree=3,
            path='testExp2_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        4: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2]*12, up=[2]*12),
            I_zones=Zones('box', low=[-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5],
                          up=[-1.4, -1.3,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5]),
            U_zones=Zones('box', low=[-0.1, 0.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5],
                          up=[0.1, 1,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5,-1.5, -1.5]),
            f=[lambda x, u: -x[0] + x[0] * x[1],
               lambda x, u: u - x[0] + 0.25 * x[1],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=1,
            degree=2,
            path='testExp3_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=3,
            k=50,
        ),
        5: Example(  ## 当前例子为展示用例
            n_obs=12,
            D_zones=Zones('box', low=[-3.15, -5]*6, up=[3.15, 5]*6),
            I_zones=Zones('box', low=[-2, -2]*6, up=[2, 2, -2,-2,-2,-2,-2,-2,-2,-2,-2,-2]),
            U_zones=Zones('box', low=[2.5, 2.5, -2,-2,-2,-2,-2,-2,-2,-2,-2,-2], up=[3, 3, -2,-2,-2,-2,-2,-2,-2,-2,-2,-2]),
            f=[lambda x, u: x[1],
               lambda x, u: - 10 * sin(x[0]) - 0.1 * x[1] + u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],

            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=1,
            path='textNonlinear0_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=2,
            k=50
        ),
        6: Example(  # 5�����ϸĵģ�����δ֪
            n_obs=12,
            D_zones=Zones('box', low=[-4, -4]*6, up=[4, 4]*6),
            I_zones=Zones('ball', center=[1, 0,0,0,0,0,0,0,0,0,0,0], r=0.2),
            U_zones=Zones('ball', center=[-1, 1,0,0,0,0,0,0,0,0,0,0], r=3.25),
            f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=1,
            degree=3,
            path='trainExp1_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            k=50,
        ),
        7: Example(  ## ��ǰ����Ϊչʾ����
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=25),
            I_zones=Zones('ball', center=[0]*12, r=0.25),
            U_zones=Zones('ball', center=[10, 10, 10, 10,0,0,0,0,0,0,0,0], r=5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2]),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],

            B=None,  ## û���ϰ�����д None
            u=0.4,
            degree=3,
            path='dim4_0_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=50,
        ),
        8: Example(
            n_obs=12,
            D_zones=Zones('ball', center=[0]* 12, r=16),
            I_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                          up=[0.2, 0.2, 0.2, 0.2]*3),
            U_zones=Zones('ball', center=[-2, -2, -2, -2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], r=2),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=1,
            degree=3,
            path='dim4_1_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=5,
            k=100,
        ),
        9: Example(
            # ����һ�����ӻ����������˲���ȫ����ͳ�ʼ����
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=16),
            I_zones=Zones('box', low=[-0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], up=[0.3, 0.3, 0.3, 0.3]*3),
            U_zones=Zones('ball', center=[-2.5, -2.5, -2.5, -2.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], r=2),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=1,
            degree=3,
            path='dim4_2_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=6,
            k=100,
        ),
        10: Example(
            # �ڵ�һ�����ӻ����������˲���ȫ����ͳ�ʼ������С����ʽ����
            n_obs=12,
            D_zones=Zones('ball', center=[0]*12, r=16),
            I_zones=Zones('ball', center=[0]*12, r=0.5),
            U_zones=Zones('ball', center=[1.5, 1.5, -1.5, -1.5, 0, 0, 0, 0, 0, 0, 0, 0], r=2.5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2]),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=0.4,
            degree=3,
            path='dim4_3_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=7,
            k=50,
        ),
        11: Example(  # �������1 �ɹ��� �������2 ʧ��
            n_obs=12,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('box', low=[-1.3] * 12, up=[1.3] * 12),
            I_zones=Zones('box', low=[-0.8,-0.8,-0.8,-0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8], up=[0.8] * 12),
            U_zones=Zones('box', low=[0.9,0.9,0.9,0.9,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
                          up=[1.3,1.3,1.3,1.3,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8]),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: 1 + sin(x[1]) * (x[1] * x[1] - cos(x[1])),
               lambda x, u: u * cos(x[1]) + x[1] * x[1] * cos(x[1]) * sin(x[1]) - 2 * sin(x[1]) / (
                           (1 + sin(x[1])) ** 2),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=0.2,
            degree=3,
            path='Cartpole_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50
        ),
        12: Example(  ## C7�������޸ģ�����δ֪
            n_obs=12,
            D_zones=Zones('ball', center=[0, 0, 0, 0]*3, r=25),
            I_zones=Zones('ball', center=[0, 0, 0, 0]*3, r=0.25),
            U_zones=Zones('ball', center=[10, 10, 10, 10,0, 0, 0, 0,0, 0, 0, 0], r=5),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2]),
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,  ## û���ϰ�����д None
            u=0.4,
            degree=3,
            path='dim4_0_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=50,
        ),
        13: Example(  ## ��ǰ����Ϊչʾ����
            n_obs=12,
            D_zones=Zones('box', low=[0]*12, up=[10]*12),
            I_zones=Zones('box', low=[3]*12, up=[3.1, 3.1, 3.1, 3.1, 3.1, 3.1,3,3,3,3,3,3]),
            U_zones=Zones('box', low=[4, 4.1, 4.2, 4.3, 4.4, 4.5,3,3,3,3,3,3], up=[4.1, 4.2, 4.3, 4.4, 4.5, 4.6,3,3,3,3,3,3]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_0_uni/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=8,
            k=100,
        ),
        14: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[1] * 12, up=[2,2,2,2,2,2,1,1,1,1,1,1]),
            U_zones=Zones('box', low=[-1,-1,-1,-1,-1,-1,1,1,1,1,1,1] , up=[-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,1,1,1,1,1,1]),
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
            degree=3,
            path='dim6_test_1_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=100,
        ),
        15: Example(  # �� 0 �Ļ����ϸı䲻��ȫ�ͳ�ʼ����
            n_obs=12,
            D_zones=Zones('box', low=[0] * 12, up=[10.0] * 12),
            I_zones=Zones('box', low=[2.0] * 12, up=[2.1,2.1,2.1,2.1,2.1,2.1,2.0,2.0,2.0,2.0,2.0,2.0]),
            U_zones=Zones('box', low=[3, 3.1, 3.2, 3.3, 3.4, 3.5,2.0,2.0,2.0,2.0,2.0,2.0],
                          up=[3.1, 3.2, 3.3, 3.4, 3.5, 3.6,2.0,2.0,2.0,2.0,2.0,2.0]),
            f=[lambda x, u: -x[0] ** 3 + 4 * x[1] ** 3 + u,
               lambda x, u: -x[0] - x[1] + x[4] ** 3,
               lambda x, u: x[0] * x[3] - x[2] + x[4] ** 3,
               lambda x, u: x[0] * x[2] + x[2] * x[5] - x[3] ** 3,
               lambda x, u: -2 * x[1] ** 3 - x[4] + x[5],
               lambda x, u: -3 * x[2] * x[3] - x[4] ** 3 - x[5],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim6_test_2_uni/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=10,
            k=100,
        ),
        16: Example(  # ��1�Ļ����ϸı��ʼ����
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0]*12, up=[1,1,1,1,1,1,0,0,0,0,0,0]),
            U_zones=Zones('box', low=[-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0], up=[-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0]),
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
            degree=3,
            path='dim6_test_3_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=100,
        ),
        17: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.99] * 12, up=[1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,1.01,0.99,0.99,0.99]),
            U_zones=Zones('box', low=[1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,0.99,0.99,0.99], up=[2,2,2,2,2,2,2,2,2,0.99,0.99,0.99]),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=2,
            path='dim9_0_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=100  # 3000
        ),
        18: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.50] * 12, up=[0.90,0.9,0.9,0.90,0.9,0.9,0.90,0.9,0.9,0.5,0.5,0.5]),
            U_zones=Zones('box', low=[1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,1.8,0.5,0.5,0.5], up=[2,2,2,2,2,2,2,2,2,0.5,0.5,0.5]),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=2,
            path='dim9_1_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=13,
            k=100  # 3000
        ),
        19: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[0.66] * 12, up=[0.90,0.9,0.9,0.90,0.9,0.9,0.90,0.9,0.9,0.66,0.66,0.66]),
            U_zones=Zones('box', low=[1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,1.9,0.66,0.66,0.66], up=[2,2,2,2,2,2,2,2,2,0.66,0.66,0.66]),
            f=[lambda x, u: 3 * x[2] + u,
               lambda x, u: x[3] - x[1] * x[5],
               lambda x, u: x[0] * x[5] - 3 * x[2],
               lambda x, u: x[1] * x[5] - x[3],
               lambda x, u: 3 * x[2] + 5 * x[0] - x[4],
               lambda x, u: 5 * x[4] + 3 * x[2] + x[3] - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
               lambda x, u: 5 * x[3] + x[1] - 0.5 * x[7],
               lambda x, u: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
               lambda x, u: 2 * x[5] * x[7] - x[8],
               lambda x, u: 0,
               lambda x, u: 0,
               lambda x, u: 0
               ],
            B=None,
            u=3,
            degree=3,
            path='dim9_2_test_uni/model',
            dense=5,
            units=30,
            activation='relu',
            id=14,
            k=100  # 3000
        ),
        20: Example(  # ��train 0�Ļ����ϸı䲻��ȫ����
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
        21: Example(  # ��0�Ļ����ϸı��ʼ����
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
    return Env(examples[i])

def get_Env1(i):
    examples = {
        0: Example(  ## 当前例子为展示用例
            n_obs=3,
            D_zones=Zones(shape='box', low=[-2.2, -2.2, -2.2], up=[2.2, 2.2, 2.2]),
            I_zones=Zones(shape='box', low=[-0.4, -0.4, -0.4], up=[0.4, 0.4, 0.4]),
            U_zones=Zones(shape='box', low=[2, 2, 2], up=[2.2, 2.2, 2.2]),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u,  ##--+
               ],
            B=None,  # lambda x: 306.5783213 + 35.3288 * x[0] - 122.5043 * x[1] + 217.9696 * x[2] - 16.8297 * x[
            #     0] ** 2 + 11.0428 * x[0] * x[1] + 39.0244 * x[0] * x[2] - 169.7252 * x[1] ** 2 - 185.8183 * x[1] * x[
            #                 2] - 29.7622 * x[2] ** 2,  ## 没有障碍函数写 None
            u=3,
            degree=3,
            path='ex0/model',
            dense=4,
            units=30,
            activation='relu',
            id=0,
            k=50,
        ),
        1: Example(
            n_obs=4,
            # D_zones=Zones('ball', center=[0, 0, 0, 0], r=10),
            # I_zones=Zones('box', low=[-1, -1, -1, -1], up=[1, 0.6, 0.5, 0.5]),
            # U_zones=Zones('ball', center=[1.5, 1.5, 0, 0], r=1),

            D_zones=Zones('ball', center=[0, 0, 0, 0], r=25),
            I_zones=Zones('ball', center=[0, 0, 0, 0], r=0.25),
            U_zones=Zones('ball', center=[1.5, 1.5, -1.5, -1.5], r=0.25),
            f=[lambda x, u: x[2],
               lambda x, u: x[3],
               lambda x, u: x[1] - 2 * x[0] + 0.1 * (-x[0] ** 3 + (x[1] - x[0]) ** 3 + x[2] - x[3]) + u,
               lambda x, u: x[0] - x[1] + 0.1 * (x[0] - x[1]) ** 3 + 0.1 * (x[3] - x[2])],
            B=None,  ## 没有障碍函数写 None
            u=0.4,
            degree=3,
            path='ex1/model',
            dense=5,
            units=30,
            activation='relu',
            id=1,
            k=50
        ),

        2: Example( #Chesi04
            n_obs=2,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('box', low=[-3, -3], up=[-1, -1]),
            U_zones=Zones('ball', center=[3, 2], r=1),
            f=[lambda x, u: -x[0] + x[1] - x[0] ** 2 - x[1] ** 3 + x[0] * u,
               lambda x, u: -2 * x[1] - x[0] ** 2 + u],
            # B=lambda x: 0.2414522721 + 2.0611 * x[0] + 1.0769 * x[1] - 0.0870 * x[0] ** 2 + 0.4085 * x[0] * x[
            #     1] + 0.2182 * x[1] ** 2,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='ex2/model',
            dense=4,
            units=20,
            activation='relu',
            id=2,
            k=50
        ),
        3: Example( #Chesi04
            n_obs=2,
            D_zones=Zones('box', low=[-2, -2], up=[2, 2]),
            I_zones=Zones('box', low=[-0.1, -0.1], up=[0, 0]),
            U_zones=Zones('box', low=[1.2, -0.1], up=[1.3, 0.1]),
            f=[lambda x, u: x[1],
               lambda x, u: -x[0] - x[1] + x[1] ** 2 + x[0] ** 2 * x[1] + u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='ex3/model',
            dense=4,
            units=20,
            activation='relu',
            id=3,
            k=50,
        ),
        4: Example( #Chesi04
            n_obs=4,
            D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
            I_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
            U_zones=Zones('ball', center=[-2, -2, -2, -2], r=1),
            f=[lambda x, u: -x[0] - x[3] + u,
               lambda x, u: x[0] - x[1] + x[0] ** 2 + u,
               lambda x, u: -x[2] + x[3] + x[1] ** 2,
               lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='ex4/model',
            dense=5,
            units=30,
            activation='relu',
            id=4,
            k=100,
        ),
        5: Example(
            n_obs=2,
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('ball', center=[1, 0], r=0.25),
            U_zones=Zones('ball', center=[-1, 1], r=0.25),
            f=[lambda x, u: -6 * x[0] * x[1] ** 2 - x[0] ** 2 * x[1] + 2 * x[1] ** 3,
               lambda x, u: x[1] * u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=1,
            degree=3,
            path='ex5/model',
            dense=4,
            units=20,
            activation='relu',
            id=5,
            k=50,
        ),
        6: Example(
            n_obs=2,
            # D_zones=Zones('box', low=[-3, -3], up=[3, 3]),
            # I_zones=Zones('ball', center=[1, 1], r=0.25),
            # U_zones=Zones('ball', center=[-1, -1], r=0.36),
            D_zones=Zones('box', low=[-4, -4], up=[4, 4]),
            I_zones=Zones('ball', center=[1, 1], r=0.5),
            U_zones=Zones('ball', center=[-2, -2], r=0.5),

            f=[lambda x, u: x[1],
               lambda x, u: -0.5 * x[0] ** 2 - x[1] + u,
               ],
            B=None,  ## 没有障碍函数写 None
            u=0.3,
            degree=3,
            path='ex6/model',
            dense=4,
            units=20,
            activation='relu',
            id=6,
            k=50
        ),
        7: Example(
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
            path='ex7/model',
            dense=5,
            units=30,
            activation='sigmoid',
            id=7,
            k=400,
        ),
        8: Example(
            n_obs=7,
            D_zones=Zones('box', low=[-2, -2, -2, -2, -2, -2, -2], up=[2, 2, 2, 2, 2, 2, 2]),
            I_zones=Zones('box', low=[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                          up=[1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]),
            U_zones=Zones('box', low=[1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8], up=[2, 2, 2, 2, 2, 2, 2]),
            f=[lambda x, u: -0.4 * x[0] + 5 * x[2] * x[3],
               lambda x, u: 0.4 * x[0] - x[1],
               lambda x, u: x[1] - 5 * x[2] * x[3],
               lambda x, u: 5 * x[4] * x[5] - 5 * x[2] * x[3],
               lambda x, u: -5 * x[4] * x[5] + 5 * x[2] * x[3],
               lambda x, u: 0.5 * x[6] - 5 * x[4] * x[5],
               lambda x, u: -0.5 * x[6] + u

               ],
            B=None,
            u=3,
            degree=3,
            path='ex8/model',
            dense=5,
            units=30,
            activation='relu',
            id=8,
            k=800
        ),
        9: Example(
            n_obs=5,
            D_zones=Zones('box', low=[-3, -3, -3, -3, -3], up=[3, 3, 3, 3, 3]),
            I_zones=Zones('ball', center=[1, 1, 1, 1, 1], r=0.25),
            U_zones=Zones('ball', center=[-2, -2, -2, -2, -2], r=0.36),
            f=[lambda x, u: -0.1 * x[0] ** 2 - 0.4 * x[0] * x[3] - x[0] + x[1] + 3 * x[2] + 0.5 * x[3],
               lambda x, u: x[1] ** 2 - 0.5 * x[1] * x[4] + x[0] + x[2],
               lambda x, u: 0.5 * x[2] ** 2 + x[0] - x[1] + 2 * x[2] + 0.1 * x[3] - 0.5 * x[4],
               lambda x, u: x[1] + 2 * x[2] + 0.1 * x[3] - 0.2 * x[4],
               lambda x, u: x[2] - 0.1 * x[3] + u
               ],
            B=None,
            u=0.5,
            degree=3,
            path='ex9/model',
            dense=5,
            units=30,
            activation='relu',
            id=9,
            k=200
        ),
        10: Example(
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
            path='ex10/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            k=50  # 500
        ),
        11: Example(
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
            degree=3,
            path='ex11/model',
            dense=5,
            units=30,
            activation='relu',
            id=11,
            k=300  # 3000
        ),
        12: Example(
            n_obs=12,
            D_zones=Zones('box', low=[-2] * 12, up=[2] * 12),
            I_zones=Zones('box', low=[-0.1] * 12, up=[0.1] * 12),
            U_zones=Zones('box', low=[0, 0, 0, 0.5, 0.5, 0.5, 0.5, -1.5, 0.5, 0.5, -1.5, 0.5],
                          up=[0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, -0.5, 1.5, 1.5, -0.5, 1.5]),
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
            path='ex12/model',
            dense=5,
            units=30,
            activation='relu',
            id=12,
            k=50  # 6000
        )
    }
    return Env(examples[i])

def get_Env(i):
    examples = {#-----原始例子-----
        0: Example(
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
        ),
        1: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.50] * 9, up=[0.90] * 9),
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
            path='dim9_1_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=13,
            k=100  # 3000
        ),
        2: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[0.88] * 9, up=[0.90] * 9),
            U_zones=Zones('box', low=[1.88] * 9, up=[2] * 9),
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
            degree=3,
            path='dim9_2_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=14,
            k=100  # 3000
        ),
        3: Example(
            n_obs=9,
            D_zones=Zones('box', low=[-2] * 9, up=[2] * 9),
            I_zones=Zones('box', low=[1.2] * 9, up=[1.45] * 9),
            U_zones=Zones('box', low=[1.88] * 9, up=[2] * 9),
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
            path='dim9_3_test/model',
            dense=5,
            units=30,
            activation='relu',
            id=15,
            k=100  # 3000
        )
    }
    return Env(examples[i])

