# Policy Gradient Notes
Information on Policy Optimization papers for methods like; REINFORCE, VPG, TRPO & PPO


## General Info
- Policy Optimization methods are compatible with Discrete and Continuous action spaces (not relegated to either)
- Directly optimize for the item we care about (policy networks learn from our estimate of the reward gradient, reward being the proficency at the task in question for the Agent, given as feedback from the Environment)


## Papers


### Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE), Williams (1992)
- Link- https://link.springer.com/content/pdf/10.1007/BF00992696.pdf
- Main Contributions: the original proposal of a Policy Gradient RL Algorithm, REINFORCE (does not use the same notion of Advantage as is used in the VPG paper and other more modern techniques Policy Gradient techniques, instead uses the discounted actual rewards from each timestep)

- Actual RL Agents are anticipated to have to integrate several techniques within the RL umbrella, or at least require us to relinquesh some of the assumptions made about the simplicity of environments and silo'd-ness of the algorithms (like the RL Agents that make it into consumer stuff, this was wrtten in 1992)
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
- John Schulman's Thesis from his PhD at Berkeley, proposes TRPO and GAE (and contains good explanations of concepts in RL, specifically related to Policy Optimization)
- Optimization in RL: Maximize total expected reward wrt Parameters of the Policy (which is a Neural Network)
- Main Contributions: Trust Region Policy Optimization (TRPO), Generalized Advantage Estimation (GAE), Unifying view of Optimizing Expectations (in RL but in several other traditional domains of Machine Learning/Optimization as well -- leading to the idea of applying RL to more prediction & probabilistic modeling problems)
  - Amazing diagrams of many of the Processes in RL (i.e, Partially-Observed vs. Fully-Observed environments)
  - Great explanation of Expectations for Stochastic Policies (see page 10)
  - Brilliant explanation of Deterministic (discrete) and Stochastic (continuous) Policies as instantiated with Neural Networks
- Deep Learning as a field reduced learning "how create a function that makes good predictions on unseen data" to optimization "minimize training error + regularization on a training set" (interesting take on DL) -- this paradigm doesn't directly translate to RL as there is no access to the meaningful data distribution (i.e, the reward function and fully-observable environment State are not made available or cannot be made available to the Agent) and our Agent's learn through forms of trial and error
- Policy Gradient methods suffer in Two Ways: 
  1. Requirement of a large number of samples required to learn an effective policy 
  2. Difficulty of obtaining Monotonically Increasing performance for a Policy, given the non-stationarity of incoming data
  - Schulman addresses these issues rather head on in Sections 4 and 3 respectively.

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
- Policy Gradients as explained here do not use the naive? or sum of expected rewards as were used in REINFORCE, instead *Advantage* and a different scheme for determining rewards for the Gradient Updates are used

- If we can sample infinitely, the difference between regular MDPs and POMDPs is blurred -- that is Partially Observable MDPs can be sampled continuously to get the data we need for function approximation (behavior is very similar to Fully-Observable MDPs) 
- The baseline used in the Reward Estimation CAN BE a State-Value function (independent of the Policy Network's foward pass function), Schulman calls this a "Near-Optimal" choice of Baseline 
  - The State-Value function as listed here is essentially the Expectation of Total Rewards for a Trajectory (subscript t) conditioned on the state at that timestep and the action as sampled from the Policy Network (see page 14 for the Math)
  - This approach builds on the original REINFORCE sum of all rewards approach as by using a baseline to scale/alter the rewards we recieve from the environment we can compute a more accurate update -- basically by removing the expected return from the reward (and later gradient update) recieved from that State-Action pair we can determine if the Action we took in that State was *better than expected* 
  - Better than expected (or worse than) gives us a more salient reward signal to update our Policy Network with (optimizing for good or better actions, not just for large reward signals)
  - This baseline reward sees the addition of a discount term at the bottom of page 15, which aims to reduce the overall variance (in exchange for a bit more bias) and lets our Value-Function estimate the *Discounted Sum of Rewards* (more accurate representation of the value of a given state)

#### 
- 

#### Generalized Advantage Estimation (GAE)
- Policy Gradient Estimator = the thing that provides our Policy Network with the Gradient/"Ground Truth" signal to update weights
  - In REINFORCE this is the sum of expected discounted rewards (1st section in this Thesis adds a Baseline & Sutton 2000 also mentions "advantage")
  - This section is about the GAE and how it can be used to reduce variance of Policy Gradient Estimators to, hopefully, reduce the number of samples required to learn an effective Policy 
- GAE is parameterized by gamma (γ) and lambda (λ) which are both floats in the range [0, 1]
  - Similar methods were proposed in previous Actor-Critic works (see page 45 for names/links)
  - This method involves two Neural Networks:
    1. Policy Network
	2. Value-Function Network
- Discounting is not in the Math of this bit because gamma is incorporated into the GAE function (discounts are apart of this specific implementation)
  - Gamma here serves (sort of) the same purpose as in simple discounted rewards, but here is intended as a *variance reduction parameter* for the *undiscounted problem*
- Bottom of Page 46 has an excellent explanation of the different Policy Gradient Estimation methods (total reward, baselined rewards, advantages, TD Residual, etc.)
- 


### Benchmarking Deep Reinforcement Learning for Continuous Control, Duan et al. 2016
- Link- https://arxiv.org/pdf/1604.06778.pdf
- Main Contributions: Code for a Benchmarking Suite for RL (tasks to test the performance of new methods on, like; Cartpole, 3D Humanoid Locomotion, partially observed environments and hierarchical structured tasks) & evaluation of existing methods + new insights from that evaluation
- These contributions (code env and baseliines) were originally released at- https://github.com/rll/rllab which now has become the project- https://github.com/rlworkgroup/garage

#### Environments
- The code (as done in the modern [garage](https://github.com/rlworkgroup/garage) repo) has support for all of the OpenAI Gym environments plus a few more
- Envs are separated into 4 Categories: 
  1. Basic Tasks: originals from the RL Landscape; cartpole, mountain car, inverted pendulum, etc. (the low-dimensional sanity checks)
  2. Locomotion Tasks: Higher dimensional tasks; walker, ant, humanoid, realistic-humanoid, etc. (to prove a new method work works)
  3. Partially Observable Tasks: preturbations to the original environment (limiting observations, adding noise/delaying actions and system identification - Agent has multiple models it needs to work with and therefore identify what it has in addition to task solving)
  4. Hierarchical Tasks: combinational tasks that require both low-level control and high-level goal direction (locomotion+get good or locomotion+reach a specific point in a maze)
  - There are 15 supported base environments plus all of the variations on those provided in the 3rd and 4th Env Categories.

#### Algorithms
- They implement a number of the SOTA algorithms at the time (2016) including methods from Batch & Online RL -- and algorithms from Policy Optimization, Evolutionary/Gradient-Free Optimization and also present Recurrent Variations of a few of the Base Methods
- To run an experiment on these algorithms they implement an interesting metric; 1/sum(i_training_iterations) * sum(number of total training iterations) * sum(number trajectories for the i-th training iteration) * R_in (the undiscounted return for the n-th trajectory of the i-th iteration) --> this leads to a reasonably fair comparison across all of the different techniques, a metric that measures performance (what we supposedly care about with testing out new methods)
  - Selecting Best Hyperparameters: to get the best hyperparams for a method, in this work the authors create a Grid Search (collection of hyperparameters across two environments per category of Environments) and run each combination of hyperparameters on those test enviornments across 5 different seeds -- the selection criterion for the *best* hyperparameters is the mean(returns) - std(returns) for each hyperparameter
- All Partially Observable tasks here are implemented with a *Recurrent Network Architecture*
- All Gradient Based algorithms here also have a baseline subtracted from their Reward Estimate (with the exception of REPS)


### High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016(b)
- Link- https://arxiv.org/pdf/1506.02438.pdf
- Main Contributions: here is where Schulman proposed Generalized Advantage Estimation (GAE) and confirms the method's usefulness with Locomotion Tasks (mujoco Ant and Humanoid) in conjunction with Trust Region Policy Optimization (TRPO)
- GAE = Variance Reduction Scheme (for Policy Optimization/Learning from the Reward Gradient)
- "Policy Gradients methods provide a way to reduce Reinforcement Learning to Stochastic Gradient Descent"
- A bit of overlap with Schulman's thesis (explanation of Policy Optimization funcs)

- Value-Functions (a mapping of a State-Action pair to the estimated value/reward for that pair) are a nice way of combating the "Credit Assignment Problem" in RL (which Actions deserve "Credit"/reinforcement for leading to a Reward? -- ultimately leading to updating the Policy in the better performing direction)
  - Value-Functions can be used in RL in a plethora of ways (Q-learning) but here the Value-Function *estimates the update* for the Parameterized Policy (pi) -- i.e, helps determine the learning/reward gradient
- The GAE is a Policy Gradient Estimator that utilizes a Value-function in it's calculation of the Policy's learning gradient
  - It trades off a small increase in bias for a larger decrease in variance (wrt the Policy's learning) -- decreasing the number of samples required to learn an effective Policy, but perhaps increasing the likelihood of falling into a Local Optima
  - This sort of approach has "seen the light of day" in the context of an Online Actor-Critic architecture (Kimura & Kobayashi, 1998; Wawrzynski, 2009) but this work is the first in the Policy Optimization space
- Policy & Value-Function here are both Neural Networks with about 10^4 Parameters (each)



