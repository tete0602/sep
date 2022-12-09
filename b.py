# encoding=utf-8
import _thread as thread

from Env2 import ex_9_dim
from polyexample import train_with_envlist_ddpg
from mamlRL import train_with_envlist_maml
import random
from multiprocessing import Process


# train_with_envlist_ddpg(test_ex_list)
# train_with_envlist_maml(test_ex_list)

def mult_process():
    process = []

    p2 = Process(train_with_envlist_maml(test_ex_list))
    p2.start()


    p1 = Process(train_with_envlist_ddpg(test_ex_list))
    p1.start()

    process.append(p2)
    process.append(p1)



if __name__ == '__main__':
    test_ex_list = []
    for i in range(2):
        test_ex_list.append(ex_9_dim(random.randint(0, 49)))
    mult_process()


