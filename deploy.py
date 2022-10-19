# Deploy an Agent
"""
this file will compose the training of an RL Agent (sample usage for importing an algorithm & making an Agent)
"""

import gym
import torch

from mi6.algorithms import VPG
from mi6.algorithms import REINFORCE as rf
from mi6.algorithms.REINFORCE import REINFORCE
from mi6.core import interaction



# Instantiate Agent & Environment
# env = gym.make('CartPole-v1')
#env = gym.make('MountainCar-v0')
#pi = REINFORCE(2, 3, learning_rate=7e-4, gamma=0.95)
env = gym.make('LunarLander-v2')
pi = REINFORCE(8, 4, learning_rate=7e-4, gamma=0.95)

#env = gym.make('Ant-v3')
#pi = REINFORCE(27, 8, learning_rate=7e-4, gamma=0.95)

score = 0.0
print_interval = 50
num_episodes = 200_000


for n_epi in range(num_episodes):
    s = env.reset()
    done = False

    while not done: # CartPole-v1 forced to terminates at 500 step.
        env.render() #human viz of training process
        
        # Give Policy Observation, Get the Sampled Action -- ORIGINAL VERSION
        # prob = pi(torch.from_numpy(s).float())
        # m = Categorical(prob) #input action probabilities into Categorical Dist, sample discrete action 
        # a = m.sample() #sample discrete action, i.e an int tensor 

        # print("probs:", prob)
        # print("action:", a)
        # print("actionitem:", a.item())
        # s_prime, r, done, info = env.step(a.item())
        # pi.append_data((r,prob[a]))

        # Get Action from Logits -- CKG Updated
        obs = torch.from_numpy(s).float() #convert state given from Env into torch vec for passing to network (observation)
        logits = pi(obs) #forward pass, computes action logits
        action, prob_a = rf.sd_sampler(logits) #sample action (int) and get corresponding probabilties

        s_prime, r, done, info = env.step(action) #interact w environment and get next state info
        pi.append_data((r, prob_a)) #append the (reward, action_probability) tuple to the Policy Data Buffer
        s = s_prime #next/new state is now the current state
        score += r #episode reward

    pi.parameter_update() #after termination of episode, update the policy

    # Print Info per N Trajectories (episodes of running a specific Policy)
    if n_epi%print_interval==0 and n_epi!=0:
        # print("Episode {}\tavg_score {}".format(n_epi, score/print_interval)) #original print statement
        rf.print_info(n_epi, print_interval, score)
        score = 0.0
env.close()
