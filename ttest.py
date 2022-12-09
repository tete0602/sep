import random
def a():
    env=[]
    env1=[]
    for i in range(2):
        a=random.randint(1,3)
        b=2
        env.append(a)
        env1.append(b)
    for i in range(2):
        print(env)
        print(env[i])
        print(env1)
        print(env1[i])

a()