# Module for Facilitating Agent Interaction with Environments


print("Interaction Module imported") #testing the module dir struct (we want non-disgusting imports)


# Util to Parse out information from Envs, pass to Algorithms to set proper dimensions
# Reference- https://www.gymlibrary.dev/api/spaces (need write logic to get proper dims for all relevant Spaces)
import gym

env = gym.make("Pendulum-v0")
env = gym.make("CartPole-v1")

obs = env.observation_space.shape
act = env.action_space.shape

print("Observation space:", obs)
print("Action space:", act)

# name this func something like;
def environment_dims(env):
    return observation_dims, action_dims #wherein each is a dict like: {"dim" : length_of_vector, "type" : continuous, ints, etc.}
                                         #and provides end user (me) with the necessary info (programatically) to connect the env and agent



# Action Mapper -- not really needed (moving over to implementing "same name" functions for each algorithm with different under the hoods)
#if there is shared code then we can move those into this file at a later date
"""
PolicyNetwork Logits --> Action Probabilities

1. Deterministic Setting (Softmax on Logits) 
2. Discrete Stochastic Setting (Torch Categorical Dist)
3. Continuous Stochastic Setting (Torch Gaussian Dist)

"""