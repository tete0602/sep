# encoding=utf-8
import _thread as thread

# from Env2 import ex_9_dim
from Env2 import ex_12_dim
# from Env2 import ex_6_dim
# from Env2 import ex_2_dim
# from Env2 import ex_4_dim
# from Env2 import ex_2_dim

from polyexample import train_with_envlist_ddpg
# from mamlRL import train_with_envlist_maml
from mamlRL2 import train_with_envlist_maml

import random
#
def get_ex_list():
    test_ex_list = []
    for i in range(1):

        test_ex_list.append(ex_12_dim(random.randint(0, 49)))
        # test_ex_list.append(ex_4_dim(random.randint(0, 49)))
        # test_ex_list.append(ex_2_dim(random.randint(0, 49)))
        # test_ex_list.append(ex_2_dim(random.randint(0, 49)))
        # test_ex_list.append(ex_9_dim(random.randint(0, 49)))
    return test_ex_list
#
def get_point_list():
    test_ex_list = get_ex_list()
    tot_list = []
    for item in test_ex_list:
        s_list = []
        for _ in range(100):
            s_list.append(item.reset())
        tot_list.append(s_list)
    return tot_list



if __name__ == '__main__':



    # locks = []
    # for _ in range(2):
    #     lock = thread.allocate_lock()
    #     lock.acquire()
    #     locks.append(lock)
    #
    # try:
    #     thread.start_new_thread(train_with_envlist_ddpg, (test_ex_list, locks[0]))
    #     locks[1].release()
    #     # thread.start_new_thread(train_with_envlist_maml, (test_ex_list, locks[1]))
    # except Exception as e:
    #     print("Error: 多线程执行失败", e)
    #
    # while (True):
    #     if not locks[0].locked() and not locks[1].locked():
    #         print("子线程执行结束，主线程退出")
    #         break

    test_ex_list = get_ex_list()
    tot_list = get_point_list()
    print('tot_list len', len(tot_list))
    print('-----开始MAML-------')
    train_with_envlist_maml(test_ex_list,tot_list)
    print('-----开始DDPG-------')
    train_with_envlist_ddpg(test_ex_list, tot_list)
