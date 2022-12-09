from Env import get_Env,get_testEnv_2dim,get_testEnv_4dim,get_testEnv_6dim,get_testEnv_9dim,get_testEnv_12dim
env = get_testEnv_12dim(1)
with open('cofe/cofe_ex{}{}.txt'.format('meta',env.id,'_with_lidao'if env.is_lidao else ''),'r') as f:
    a=f.read().split('\n')

s=''
for v in a[:-1]:
    x,y=v.split(':')
    x=x.replace(' ','')
    if x=='1':
        s+='('+y+')'
    else:
        s+='('+y+')'+'*'+x
    s+='+'

for i in range(13,-1,-1):
    #print(i,i+1)
    s = s.replace('x{}'.format(i),'x{}'.format(i+1))

for i in range(20,25):
    #print(i,i+1)
    s = s.replace('x{}'.format(i),'x{}'.format(i-10))

s = s.replace('+(-','-(')[:-1]
print(s)
print('u∈[-{},{}]'.format(env.u,env.u))
print('n_obs:',env.n_obs)
print('ID:',env.id)