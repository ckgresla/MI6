# REINFORCE (original version of the VPG w/o the Advantage)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Policy for Algorithm
class REINFORCE(nn.Module):
    def __init__(self, observation_dim=4, action_dim=2, learning_rate=None, gamma=None):
        super(REINFORCE, self).__init__()
        
        # Algorithm-Specific Info
        self.data = []
        self.continuous_ok = True
        self.discrete_ok = True

        # Default Hyperparameters
        self.learning_rate = 0.0002 if learning_rate == None else learning_rate
        self.gamma = 0.98 if gamma == None else gamma

        # Input/Output Dimensions for Policy Network (change per environment, default is set to "Cartpole-v1" values)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Policy Network Architecture (given appropriate observation/action | input/output dimensions)
        self.fc1 = nn.Linear(self.observation_dim, 256)
        self.fc2 = nn.Linear(256, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        Reward = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            Reward = r + self.gamma * Reward
            loss = -torch.log(prob) * Reward 
            loss.backward()
        self.optimizer.step()
        self.data = []


# Run Algorithm on Cartpole as Example
def cartpole_test():
    import gym

    env = gym.make('CartPole-v1')
    pi = REINFORCE(4, 2, learning_rate=7e-4, gamma=0.95)
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
            print("Episode {}\tavg_score {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()


# Sample Run of Algorithm
if __name__ == '__main__':
    cartpole_test()