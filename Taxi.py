# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:15:32 2020

@author: ziyac
"""
import  gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time
env=gym.make("Taxi-v3").env

# q-table
q_table=np.zeros([env.observation_space.n,env.action_space.n])

#hyper-parameter
alpha=0.1
gama=0.9
epsilon=0.1

#platting metric
reward_list=[]
dropout_list=[]
episode_number=1000
for i in range(1,episode_number):
    #initilize enviroment
    state=env.reset()
    reward_count=0
    dropouts=0
    
    while True:
     
        # exploit vs explore to find action
        if random.uniform(0,1) < epsilon:
            action=env.action_space.sample()
        else:
            action=np.argmax(q_table[state])
            
        # aciton process and reward /observation
        next_state,reward,done,_=env.step(action)
        old_value=q_table[state,action]
        next_max=np.max(q_table[next_state])
       
        
        # q-learning
        next_value=(1-alpha)*old_value+alpha*(reward+gama*next_max)
        #update q-table 
        q_table[state,action]=next_value
        state=next_state
        
        #find wrong dropouts
       
        if reward== -10:
            dropouts+=1
           
        reward_count += reward            
    
        if done:
            break
      #  env.render()
        
    if i%10 == 0:    
        dropout_list.append(dropouts)
        reward_list.append(reward_count)
        env.render()
        print("Episode: {},reward {},wrong dropout {}".format(i,reward_count,dropouts))
        #time.sleep(1)
    
    # %% visualize
fig ,axs=plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")
    
axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")
axs[0].grid(True)
axs[1].grid(True)
plt.show()
    