# do we make one of these that all algorithms import for CLI running? or do we just write a specific test function in each file...
import gym
from ..core import interaction


def cartpole_test(policy):
    
    env = gym.make('CartPole-v1')
    #env = gym.make("Pendulum-v1")
    pi = policy(4, 2, learning_rate=7e-4, gamma=0.92)
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            # print("probs:", prob)
            # print("action:", a)
            # print("actionitem:", a.item())
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        # Print Info per N Trajectories (episodes of running a specific Policy)
        if n_epi%print_interval==0 and n_epi!=0:
            print("Episode {}\tavg_score {}\tlr {}".format(n_epi, score/print_interval, pi.learning_rate))
            score = 0.0
    env.close()