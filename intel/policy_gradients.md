# Policy Gradient Notes
From the recommended background papers for VPG/Reinforce from SpinningUp + other Policy Algortihms




## Papers

### Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE), Williams (1992)
- Link- https://link.springer.com/content/pdf/10.1007/BF00992696.pdf
- Main Contributions: the original proposal of a Policy Gradient RL Algorithm, REINFORCE (does not use the same notion of Advantage as is used in the VPG paper and other more modern techniques Policy Gradient techniques)

- Actual RL Agents are anticipated to have to integrate several techniques within the RL umbrella, or at least require us to relinquesh some of the assumptions made about the simplicity of environments and silo'd-ness of the algorithms 
- Associative Tasks: in this paper are defined as Tasks in which the Learner (policy or Agent that takes observations and outputs actions) is required to learn/do an input-->output mapping, wherein the learning signal comes from only immediate rewards, referred to as "Immediate Reinforcement" (differing a bit from the concept of "Delayed Reinforcement", where reward feedback is not immediately provided to the agent)
- Search is effectively random wrt the Learner (unlike some other RL methods, mentioning A_star search, Nilsson 1980 and adaptive control-like algorithms, Bandits for example which search but also start to exploit based on the current estimate of value)
- EXPLICIT Computation of the Reward Gradient is not done, it is an *approximation* of the actual Reward signal, this still suffices to train the REINFORCE algorithm though
- "Non-Model Based" in this paper refers to the fact that they do not compute an explicit gradient of the reward function, and estimate it (models in their context presumably refer to Value function-esque systems) 
- The Approximation for the Performance Criterion (Reward function) is the Expectation of Reward conditioned on the Weight matrix (i.e, E(r | W))
- REINFORCE stands for- REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility 
  - which is the Algorithm's update function; gradient_w_ij = alpha_ij x (r - b_ij) x e_ij) -- wherein: 
    - alpha is the learning rate
    - gradient_w_ij is the update to the parameter w_ij in the network
	- r is the expected reward
	- b_ij is a "reinforcement baseline" ???
	- e_ij = ln(g/w_ij) where g is the Bernoulli Unit or some other unit definition (as specified in this paper)
  - In natural language: "the average update vector in weight space lies in a direction for which this performance measure is increasing"
- Sutton, 1984's algorithm with the *Reinforcement Comparison* update rule, is a special case of the examples provided in that it does not have a baseline (b_ij) = 0, in the weight update function. Instead it makes an "adaptive estimate" of upcoming rewards given prior experience

- Episodic REINFORCE Algorithms are those in which we account for time with termporal credit-assignment (reward is allocated based on the usefullness of actions at a particular timestep in the Episode -- different from the "timeless" approach in the above)
- This is much more in line with the available implementations of VPG/REINFORCE on the web, we save information from each episode and then backprop the reward signal at the end of the episode through the Policy Network (after discounting the rewards)
- Accumulation of Reward: to handle the timestep scenario, we can replace "r" with "sum(r_t)" over K timesteps (sum of reward up to a point)

- Convergence Properties
  - Williams was not clear (at time of writing) about the asymptotic properties of the REINFORCE Algorithms (whether or not they were guaranteed to converge to a local/global optima or if they do not converge at all) -- he does make the point however that convergence is non-essential
  - They make this claim as the outperformed a number of other SOTA algorithms at the time with this System, proving the usefulness literally
  - Attaining good solutions with REINFORCE requires a careful crafting of the Reinforcement Function (part of learing rule) -- bad choice of reward signal and as one might expect you learn the incorrect thing
  - Performance across the board is slow, even in the case of learning something useful (particularly Episodic REINFORCE! due likely to teh termporal credit-assignment diluting the updates across all weights uniformly, at least as done in this work) 
  - Williams' Statement on REINFORCE Convergence: Depending on the choice of reinforcement baseline used, any such algorithm is more or less likely to converge to a local maximum of the expected reinforcement function, with some nonzero (but typically comfortably small) probability of convergence to other points that lead to zero variance in network behavior.

- Reinforcement Baseline Choices
  - No basis for choosing among candidate Reinforcement Baselines is provided or found in this work
  - Comparison strategies seem to be better for speeding up convergence (think advantage) -- and not just faster but also better overall performance
  - One plausible method is to minimize the variance of individual weight changes as time goes on (adaptive learning rate?)


### Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al. 2000
- Link- https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
- Main Contributions: introduction of VPG and the use of an advantage function for a Policy-Gradient system

- Directly approximating the Value function (function that computes value for a given State-Action pair) was intractable
  - This is an alternate approach wherein the Policy is not extracted from the Value function, instead it is an independent piece that is updated according to *the gradient of expected reward* wrt the Policy Parameters (expected reward is what we backprop the Policy, pi, on)
  - It builds on the REINFORCE paper, specifically this method uses an *approximate* Action-Value function, the Advantage Function, that makes learning the Policy via a Gradient feasible -- something not previously tractable
  - Effectively, this combination of methods the door for full-stack backprop learning of a suitable Reward function
- Why Policy Gradients might be a better choice than Value-Functions:
  - Value-Function based RL solutions to tasks are *oriented* to be deterministic, meaning that going after a pure Value function based approach rules out or makes difficult incorporating any Stochasticity into the Policy (the value for a state is deterministic, given the current value function, in practice there may not always be a "perfect" action for a given state but a set of probabilities over actions that when sampled leads to better overall performance than the performance of a purely deterministic, Value-function based approach)
  - Value-functions are also less robust to small changes, if we nudge one State-Action pair's value down slightly, it doesn't get picked -- degrading performance if sometimes it needs to get picked despite not being as valuable as initally believed

Differences in Methods:
- Value-Functions: Approximate a Value Function (Q-table or representation of Action-value mappings) --> Compute Deterministic Policy
- VPG: Approximate a Stochastic Policy <-- Directly with the Policy Network (pi)

Why are these differences useful?
- Moving over changes from the Value Function (prior work) to the Policy Network (VPG) means that changes in the Policy/Action Determining Mechanism (change in value for a state-action pair or update of policy parameters) results in less of a dropoff for mistakenly high-value actions, under the prior approach they would not get selected, in the VPG Policy Network approach we just reduce the Action Selection probability a little bit, with backprop and enough data we can conceivably learn a useful policy that is more robust than a Q-Table of static values

- The REINFORCE algorithm's main drawback is that it is much slower to learn an optimal-ish policy than the Value-Function based approaches.
  - VPG looks to build upon this work with the addition of the "Learned Value Function" -- *approximation of true Value-Function w Advantage*

#### Policy Gradient Theorem w Advantage 
- They use the typical MDP-style formulation of the RL problem 
- Two Approaches to Formulating an Agent's Objective Function
  1. Average Reward: different starting states, train Policy to maximize reward across all of these different starting states
  2. Designated Starting State Reward: hold the Starting state s_0 fixed, train Policy to maximize reward from the same start state 
- Using the actual returns (for a full episode) as the reward to train the Policy with is the approach outlined in the REINFORCE paper
  - The addition of Value-function Approximation via Advantage separates out VPG from REINFORCE
- The advantage function is basically the difference in reward between the Mean Value of Rewards and the Value associated with the particular State (and Policy)
  - This comes about as a consequence for their derivation of the Approximation for the Value-Function, one requirement being that there be a zero mean value for each State (differences in rewards, from the zero reward, can be thought of as the "relative advantage" for each State)
  - The convergence requirement is to get the relative value of actions correct for each state; not the absolute value for actions nor the differences in value from state to state

#### REINFORCE's Reward Function
- Recall that our Policy is just a mapping of States to Action logits (which get converted into probabilities via a Softmax of our last output layer) 
- This Policy needs to be trained, that is accomplished by creating a loss function to backprop off of. For this method, we can use the actual finite-horizon or discounted infinite-horizon Rewards. The values for Rewards across all timesteps, (i.e, sum(gamma_t * Reward_t) for all timesteps t). This discounted reward function is referred to as G_t, or the discounted reward. 
  - Gamma here refers to the discount factor that exponentially decays) are multiplied with the Policy's Action proabilities (log probs for better training) and averaged over time (i.e, -1/t * sum(log(G_t * pi_t))), this gives us our Gradient that we then backprop to update the weights to be closer to the optimal policy
- Loss Computing, Backpropagation and the Optimization Step are computed at the end of a Batch (similar to regular dl) w several batches per episode
- Reward typically doesn't monotonically increase, this is to be expected


### Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs, Schulman 2016(a) 
- Link- http://joschu.net/docs/thesis.pdf 
- John Schulman's Thesis from his PhD at Berkeley, proposes TRPO and GAE
- Optimization in RL: Maximize total expected reward wrt Parameters of the Policy (which is a Neural Network)
- Main Contributions: Trust Region Policy Optimization (TRPO), Generalized Advantage Estimation (GAE), Unifying view of Optimizing Expectations (in RL but in several other traditional domains of Machine Learning/Optimization as well -- leading to the idea of applying RL to more prediction & probabilistic modeling problems)
  - Amazing diagrams of many of the Processes in RL (i.e, Partially-Observed vs. Fully-Observed environments)
  - Great explanation of Expectations for Stochastic Policies (see page 10)
  - Brilliant explanation of Deterministic (discrete) and Stochastic (continuous) Policies as instantiated with Neural Networks
- Deep Learning as a field reduced learning "how create a function that makes good predictions on unseen data" to optimization "minimize training error + regularization on a training set" (interesting take on DL) -- this paradigm doesn't directly translate to RL as there is no access to the meaningful data distribution (i.e, the reward function and fully-observable environment State are not made available or cannot be made available to the Agent) and our Agent's learn through forms of trial and error

Difference in Problem Solving -- DL vs RL
- Deep Learning: 
  - Learn a useful mapping of Inputs --> Outputs (w a Neural Network)
  - This means we approximate the function that typically does this mapping (pixel distribution of image is cat, next word in sequence is "wow", etc.)
  - Input data does not depend on prediction from model 
- Reinforcement Learning: 
  - What to approximate? We can/want approximations for Policies, Value Functions, Dynamics Models or combinations of these
  - 2 Choices to Make:
    1. What Objective to Optimize? approximate the policy, value function or dynamic model
	2. How approximate it? With a neural network or with what tool? 
  - Model's Observations (input data) are or can be dependent on the Actions (model outputs)

- Schulman's work here basically turns RL into DL, in that the Policy Gradient methods. 
  - Repeatedly Compute a Noisy Estimate of the "Gradient of Performance" () and feed that to the Policy's via SGD (to update weights)
- To combat the *Credit Assignment Problem*, Schulman here proposes GAE -- this method is a way to reduce the variance of Policy Gradient Estimation (reward that we backprop our Policy with) to make learning converge more stably & reliably

- If we can sample infinitely, the difference between regular MDPs and POMDPs is blurred -- that is Partially Observable MDPs can be sampled continuously to get the data we need for function approximation (behavior is very similar to Fully-Observable MDPs) 
- LEFT OFF ON PAGE 18

