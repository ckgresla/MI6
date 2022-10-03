# REINFORCE (original version of the Policy Gradient w/o GAE)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical




# Policy for Algorithm
class REINFORCE(nn.Module):
    """
    REINFORCE algorithm (Policy Optimization via Full Estimated Reward) as created in:
        Williams (1992)- https://link.springer.com/content/pdf/10.1007/BF00992696.pdf

    Parameters:
    - observation_dim: int, number of observations recieved from Environment
    - action_dim: int, number of actions Policy/Agent can take in Environment
    - network_architecture: dict, specification for the structure (layers, activations, optimizer, etc.) of the Policy Network
            Example: {"fc1" : 256, "fc2" : 256, "optimizer" = }  <-- figure out how make (maybe make this a function we call instead?)
    - learning_rate: float, learning rate for Policy Network Optimizer
    - gamma: float, discount rate for reward function
    """
    def __init__(self, observation_dim=4, action_dim=2, network_architecture=None, learning_rate=None, gamma=None):
        super(REINFORCE, self).__init__()
        
        # Algorithm-Environment Info (what types of environments does it work with)
        self.continuous_ok = True
        self.discrete_ok = True

        # Default Hyperparameters
        self.learning_rate = 0.0002 if learning_rate == None else learning_rate
        self.gamma = 0.98 if gamma == None else gamma

        # Input/Output Dimensions for Policy Network (change per environment, default is set to "Cartpole-v1" values)
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Data Buffer for Episodes
        self.data = [] #list of (Reward, Action) tuples per timestep t (cleared out after updating policy network params, per episode, for tracking next episode Policy performance)

        # Policy Network Architecture (given appropriate observation/action | input/output dimensions)
        if network_architecture == None:
            self.fc1 = nn.Linear(self.observation_dim, 256)
            self.fc2 = nn.Linear(256, self.action_dim)
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            # Need figure out how do nice assemblies of network
            for specification in network_architecture:
                self.specificiation = network_architecture[specification]

    # Forward Pass (given observation, matmul & activate through the network)
    def forward(self, x):
        x = F.relu(self.fc1(x)) #original Activation in minRL
        # x = torch.tanh(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=0) #original Final Layer (no go as softmax can damage training)
        x = self.fc2(x) #no softmax in the forward pass, want return logits! (instead of probs)
        return x #action_logits (log probabilities, can turn into ACTUAL probabilities with softmax)

    def append_data(self, item):
        """
        Add additional data tuples to Algorithm's Buffer
        (used for Loss estimate at end of each episode, i.e Gradient Updates)
        """
        self.data.append(item)

    def reset_data(self):
        """
        Reset Data store after updating Policy (new policy means we track new buffer of data)
        """
        self.data = [] #reset Data buffer after weight update (running a new Policy)

    # Base Reward Gradient Estimate (w/o advantage as done in VPG)
    def reward_function(self):
        """
        Estimate of the "Reward Gradient" per Timestep t in Data Buffer (updating loss for Backprop of Policy Network)
        (using the "self.data" struct)
        r_t & prob_t refer to the Reward and Action Probabilities at Timestep T
        """
        discounted_reward = 0 #per episode tracker of discounted rewards

        # This is a different way to implement "Reward-To-Go" as mentioned in- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you
        # Iterate over Data Tuples in reverse order (from last-to-first, episode-by-episode iteration, calculating discounted rewards)
        for r_t, prob_t in self.data[::-1]:
            discounted_reward = r_t + self.gamma * discounted_reward #for each timestep in episode, discount prev reward in LOOP iteration (t+1 timestep) and add that discounted sum of rewards to the current reward (gets discounted in the next loop iteration)
            loss = -torch.log(prob_t) * discounted_reward #Estimated Gradient (for the timestep in question) = -logprobs * discounted_reward
            loss.backward() #add gradients to each weight in network, parameter update happens in batch, after reward_gradient for the whole Episode is calculated 

    def parameter_update(self):
        """
        Update the Policy Network parameters:
          1. Clear out previous Gradient Values (associated with current network parameters after prev step)
          2. Calculate new Gradients for Episode (estimation of True Gradient, via the reward_function)
          3. Backprop the estimated gradient to the current parameters (update step)
          4. Clear out the data buffer (new episode with the updated policy network) -- "on-policy" so need match Policy Net actions with correctly associated data
        """
        self.optimizer.zero_grad() #clear out prev gradient
        self.reward_function() #calc gradients for prev Episode (with data buffer)
        self.optimizer.step() #backpropagation/update params as per current Gradient
        self.reset_data() #clear out data buffer for next Episode




# this function needs to be moved into the `reports` util in the "core" dir
def print_info(n_episodes, print_interval, score=0):
    """
    standardized string for printing training information (handle when to log/print logic in main script)
    how incorporate different metrics here?
    """
    avg_score = f"{score/print_interval:.2f}"
    print(f"Episode {n_episodes: ^{10}}", end="    ")
    # cp(f"Avg Score {avg_score: ^{10}}", color="red") #need import color print from the "core.reports" module
    print(f"Avg Score {avg_score: ^{10}}")
    return

# implement here for now, potentially move into interaction later (if feels better there)
def sd_sampler(action_logits):
    """
    Stochastic Discrete Action Sample Wrapper
    - Samples Actions & corresponding Probability from the PyTorch Categorical/Multinomial Distribution- https://pytorch.org/docs/stable/distributions.html#distribution

    input is the RAW logits from the output of the network (computes probabilities with Softmax under the hood)
        return is the; int for the action sampled & the Probability associated with that action
    """
    probs = torch.nn.functional.softmax(action_logits, dim=0) #convert Network outputted logits into Probabilties
    action_distribution = Categorical(probs=probs) #instantiate distribution with action probabilities (not logits, although COULD do that too)
    action = action_distribution.sample() #returns an int for the sampled action
    action_prob = probs[action] #likelihood of making this action as per the policy

    return action.item(), action_prob



# Run Algorithm on Cartpole as Example
def cartpole_test(num_episodes=1000):
    import gym

    env = gym.make('CartPole-v1')
    pi = REINFORCE(4, 2, learning_rate=7e-4, gamma=0.95)

    score = 0.0
    print_interval = 20


    for n_epi in range(num_episodes):
        s = env.reset()
        done = False

        while not done: # CartPole-v1 forced to terminates at 500 step.
            #env.render() #human viz of training process
            
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
            action, prob_a = sd_sampler(logits) #sample action (int) and get corresponding probabilties

            s_prime, r, done, info = env.step(action) #interact w environment and get next state info
            pi.append_data((r, prob_a)) #append the (reward, action_probability) tuple to the Policy Data Buffer
            s = s_prime #next/new state is now the current state
            score += r #episode reward

        pi.parameter_update() #after termination of episode, update the policy

        # Print Info per N Trajectories (episodes of running a specific Policy)
        if n_epi%print_interval==0 and n_epi!=0:
            # print("Episode {}\tavg_score {}".format(n_epi, score/print_interval)) #original print statement
            print_info(n_epi, print_interval, score)
            score = 0.0
    env.close()


# Sample Run of Algorithm
if __name__ == '__main__':
    cartpole_test()

