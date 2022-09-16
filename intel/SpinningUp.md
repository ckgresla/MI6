
# OpenAI's SpinningUp 
some notes from the documentation over at- https://spinningup.openai.com/en/latest/index.html




##  General Notes
- Expected Return for a State-Action pair is the Value for that pairing (the reward for a State-Action combo is estimated to pick the most beneficial action)
- Seed setting is Huge in RL, these make or break performance and determining whether or not an Algorithm is good requires statistically validity (i.e. significant performance on many different Seeds)
- Playing w an Agent or Seeing performance in an single env is cool, but any real evaluation of a system should be done via an Experiment (many agents with different seeds on a single env or ideally many enviroments)
- Main Characters in the RL Story; the Agent and the Environment
  - Agent is the Algorithm that we program, recieves an Observation (partial or all-encompassing) or State and can take Actions that lead to Reward Signals
  - Environment is the space the Agent exists in and is responsible for providing the Observation and Reward signals, recieves the Actions from the Agent
- Stochastic Gaussian Policies Output Log Standard Deviations instead of Regular Standard Deviations; this distinction is useful because log std devs can take any value between (-∞, ∞) while regular std devs must be non-negative, parameters are easier to train if we do not have to enforce a non-negative constraint and therefore learning the Policy becomes easier -- regular Standard Deviations can still be obtained from the log std dev by a simple exponentiation (i.e e^(σ) with the log standard deviation as σ)
- Optimal Q-Function and The Optimal Action: The Optimal Action-Value Function (i.e Q_star(s, a)) and Optimal Policy (V_star(s)) which outputs an Action, that is because the Optimal Value-Function determines Expected Rewards for Actions A in the Current State S and is also Optimal we can take the Argmax (wrt Value output) of the Actions, the Optimal Action -- if there are several "Optimal Actions" that we can get from our Action-Value Function, we can take a random choice from that set but there is a "Deterministic" Optimal Policy that selects the Best action
- Policy Entropy is a Good Measure of Learning, better than trends in Episode Rewards- http://amid.fish/reproducing-deep-rl (about halfway down in this)
- Key Papers in RL- https://spinningup.openai.com/en/latest/spinningup/keypapers.html (great list to peep, particularly Section 13 with Classics)




## Specific Concepts
States & Observations
- States are *complete* descriptions of the World
- Observations are *partial* descriptions of the state (do not contain all of the information to fully describe an Environment)
- If an Agent has access to the State, then that Environment is "Fully-Observable"; If the Agent does not have access to the State and has access only to an Observation, then that Environment is "Partially-Observable"
  - In the literature this terminology is a bit shifty, in the sense that some people call the *Observation* the *State* or vice-versa (Agent makes an Action given the Current State; it would be more appropriate here to say "Observation" but we run with it since this is technically the “State” that the Agent has access to — this is RL not Philosophy or English, leave the debating to the Logic Programs)
- Almost always a Vector, Matrix or Tensor in RL


Action Spaces
- Actions are the things an Agent can do in an environment; the Set of all possible Actions an Agent can take is referred to that Agent or Environment’s "Action Space"
- These Spaces can be either *Discrete* or *Continuous*:
  - Discrete: an Agent can only make a finite number of actions (go left or right @ some set rate; i.e. move left 1 box)
  - Continuous: an Agent’s actions are effectively infinite (move your leg 0.12 ft forward and 0.001 ft left) -- these actions can be bounded continuous values (any value within a specific range, still continuous but a bounded variation) 
  - This distinction in Action Values of an Action Space can rule out whole families of Algorithms (and need significant work to port over to an environment they are not conducive for, if they CAN be ported over)


Policies
- A Policy is a rule used by an Agent to decide what Actions to take, these Policies (figuring out what action to take given state) are either; *Deterministic* or *Stochastic* -- that means they are either real-valued (given states) or probabilistic (conditioned on the state given)
- The Policy is basically the Brain of the Agent and because the Policy is what determines the Actions to be taken it is appropriate to say something like: "The policy is attempting to maximize reward" where "policy" is interchangable with "Agent"
- These Policies in Deep RL are Parameterized Policies (Neural Networks or Models with Parameters/Weights that can be learned) -- Computable Functions that have parameters which we can adjust to update or change the behavior of the Agent (parameters of a Policy are typically denoted with the Greek; Theta (θ) or Phi (Φ) characters)
- Deterministic Policies: basically the typical FFN that outputs a Vector with logits for each respective action (we take the Max always, these logits after softmax-ing are the Action probabiltiies but we do not sample here & that is why it is Deterministic)
- Stochastic Policies: these can be broken down into two further sub-classes; *Categorical Policies* and *Diagonal Gaussian Policies* -- but in essence these are Policies that involve random variables or probabilities and are not deterministically computable outputs of a function, but are expected outputs of a function conditioned on the data 
  - Categorical Policies: are used in *discrete* action spaces and stem from the Categorical Distribution- https://en.wikipedia.org/wiki/Categorical_distribution
    - This distribution is also referred to as the "Generalized Bernoulli Distrbution" or "Mulinoulli Distribution" -- basically if we have K actions available to us, the sum of the probabilities for all of these actions must sum to 1 and the values for each action must be in the range of [0, 1]
	- The Categorical Distribution is the Generalization of the original Bernoulli Distribution (binary coin-flip) for a Categorical Random Variable (think dice roll) with more than two random outcomes (hence, Generalization) -- that being said, the Categorical Distribution is also a SPECIAL case ofthe Multinomial Distribution, as it gives the probabilities of Outcomes for a single drawing (single dice roll) rather than the expected probability over *multiple* dice rolls
  - Diagonal Gaussian Policies: are used in *continuous* action spaces and stem from the Multivariate normal Distribution- https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	- This also goes by the name of "Multivariate Gaussian Distribution" or "Joint Normal Distribution" and all of these are Generalizations of the old school, one-dimensional Gaussian (μ, σ) -- mu & sigma, to multiple dimensions. 
	- It gets it's stock from the Multivariate Central Limit Theorem, and the Multivariate Normal Distribution is able to "approximate any set of correlated real-valued random variables which cluster around a mean value"
	- It is the Generalized form of the Distribution because the former scalars, i.e (μ, σ) are replace with a Mean Vector (mu) and a Covariance Matrix Σ (capitalcase sigma) -- the "Diagonal Gaussian Distribution" is a special case of distribution as the Covariance Matrix only has entries on the Diagonal (indicies on the diagonal, i.e (i, j) where i=j) -- this Diagonal part of the Matrix is the only part of that we care about, so Sigma can be represented as a Vector of these diagonal values (as opposed to the full covariance matrix). 
- Key Computations: what we care about wrt these policies is being able to *Sample Actions* from them & Compute the *Log-Likelihoods* for particular actions (log probabilities, get an idea of "confidence" in a Model's action outputs)
  - Categorical Computations: a Categorical Policy is effectively a multi-class classifier (input_vector is the Observation, Network has some architecture that should be conducive to the task; conn layers, hidden layers, etc., and a final output layer that gets plugged into a Softmax Activation to convert the final logits into Action Probabilities)
	1. Sampling: given the Probabilities of all Actions we sample according to the Categorical Distribution with something like- https://pytorch.org/docs/stable/distributions.html#categorical -- this is an added function to the *Deterministic* Policy sampling (we would just stop at Softmax action output) that changes our deterministic output into a probabilistic one 
	2. Log-Likelihood: The output layer of out Policy is a vector that has as many values as we have actions (output_vector) -- we can treat the actions as indicies for each vector (output_layer[2] corresponds to  3, etc.) -- log-likelihood then is the same as in a regular neural network (we index the output vector for the action, i.e output_vec[1] for action 2, we want to examine and the output logits or softmax probabilities gives us the log-likelihood of the Action of Interest)
  - Diagonal Gaussian Computations: given our mu and sigma vectors, we use a Neural Network to map Observations to MEAN Actions (average Actions, not being a dickhead) -- this leads to two ways in which a covariance matrix can be represented, both represent the values still as a Vector; the *first* way is to have a single vector of log standard deviations (log σ) which is not a function of the current State (standalone standard deviation params) or the *second* way of creating a separate Neural Network that maps States/Observations to Log Standard Deviation (as an output vector given state as input) this may or may not share layers with the original net (could be a separate NN or some Frankenstein-y combination of shared layers) -- see top for note on why Log Standard Deviation instead of Regular Standard Deviation
	1. Sampling: we take our Mean Action (mu), Standard Deviation (sigma) and a Noise Vector (z) -- we can sample a value mathematically by adding the output of the Mean Vector to the Elementwise Vector Product of the sigma and z vectors and then selecting the highest probability Action -- the Noise vector is random noise that can be acquired through a [torch.normal](https://pytorch.org/docs/stable/torch.html#torch.normal) call (returns a tensor of specified shape from a normal distribution, given specified mean and standard deviation) or we could create a Normal Distribution Object that does the math to sample an Action for us, like in [torch.distributions.normal.Normal](https://pytorch.org/docs/stable/distributions.html#normal) (this takes in the mu and sigma that we would use to sample an action mathematically and sees to the error term under the hood) -- the Distribution Approach can also calculate log-likelihoods for us
	2. Log-Likelihood: mathematically we can calculate the log-likelihoods of a Diagonal Gaussian by this [formula](https://spinningup.openai.com/en/latest/_images/math/26f82323a4055444b30fa791238ec90913a12d7b.svg) which is basically a summation of the vairance across all action values times the log likelihood and pi (spherical gaussian)


Trajectories 
- The sequence of States and Actions that occur is referred to the Trajectory - The starting State of the Trajectory is s0, and this initial state is typically randomly sampled from the State-Start Distribution (referred to as p0) -- basically where the Agent begins a particular sequence/episode
- State Transitions: are what happen to the world at a timestep t in between s_t and s_t+1 (the current state and the next state after timestep t) these transitions are based on the laws of the environment and the prior step's action -- they can also come in a *Deterministic* or *Stochastic* fashion-- Actions always come from the Agent's Policy  
- Trajectories can also go by the names; "Episodes" or "Rollouts"


Reward & Return
- The Reward Function is perhaps the most critical item in RL, it depends on the *Current State of the World*, the *Action Just Taken* and the *Next State of the World*. These are the inputs into a *Reward Function* (R) and reward would look something like; r_t = R(s_t, a_t, s_t+1) where t is the current timestep. (although atypical, the input to this reward function can be shorted to take just the current state (s_t) or the current State-Action pair (s_t, a_t)) 
- Ultimately an Agent's Goal is to **maximize reward** (cumulative reward!) over a *Trajectory* (single episode of existence) -- returns come in different flavors and we have:
  - Finite-Horizon Undiscounted Return: which is the sum of rewards (returns at timesteps or r_t) over a fixed window of [steps](https://spinningup.openai.com/en/latest/_images/math/b2466507811fc9b9cbe2a0a51fd36034e16f2780.svg). 
  - Infinite-Horizon Discounted Return: which works out to be the sum of all rewards EVER obtained by an Agent, discounted by a term, typically gamma . Rewards that come further in the future are Discounted more than Near-Future rewards (as there is more ambiguity about those long-term rewards) -- Mathematically the choice of including the discount term helps if there is no Finite Value for Reward (which can break the equations, with a discount factor this can work out more easily, see- https://en.wikipedia.org/wiki/Convergent_series for more information on why Mathematically this works, jist is that we may not converge on a Reward Value in a truly infinite Reward Horizon, so Discounting makes rewards too far way inconsequential)
- The difference between the two Returns in the above can be blurred, for example: Algorithms are set to optimized Undiscounted Return but Value Functions incorporate Discount Factors into their estimation of value for a State-Action pair. 


The RL Problem
- Fundamentally, Reinforcement Learning is about Maximizing Reward (be it a Finite or Infinite-Discounted measure) by selecting a Policy that when applied by an Agent maximizes *Expected Return* across Trajectories. (expected because environments may have Stochasticity baked in to the State Transitions that are non-incorporatable to the Policy, at least not entirely in their full resolution)
- The Central Optimization Problem boils down to Selecting the Policy (pi) that maximizes *expected reward*, which approximates the *optimal policy* (pi_star)
- Expected Return is typically represented mathematically as; J(pi) (where pi is a Policy) and the Central Optimization Problem looks like; pi_star = argmax J(pi) (take the Policy that maximizes Expected Return, this approximates or is equivalent to the Optimal Policy)


Value Functions
- Knowing how "valuable" a State or State-Action pair is useful for determining the optimal policy (we can select the Actions that maximize Value in all S or S-A pairs which would maximize our rewards) -- Value then is the Expected Return for a given State with the assumption that we will hold our Policy constant for all subsequent actions.
- There are 4 Main Value Functions, check out the equations [here](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions)
  1. On-Policy Value Function: this gives the expected return if you start in State S and act (without deviation) according to Policy pi
  2. On-Policy Action-Value Function: expected return in State S given that you take arbitrary Action A (which may or may not come from the Policy pi) and then act without deviation according to the Policy after that
  3. Optimal Value Function: the expected return if you start in State S and always act according to the Optimal Policy (pi_star) in the Environment 
  4. Optimal Action-Value Function: the expected return if you start in State S and take an arbitrary Action A and then forever act according to the Optimal Policy (pi_star) in the Environment
- The reason for the "without deviation" specification in the Value Functions above is due to the notion of expected rewards, these expectations assume that Behaviour will be fixed (i.e the Policy remains consistent) -- reference the section on Behavior and Update Policies below
- If Time Dependence is not explicitly stated (wrt a Value Function's calculation) then it is typically safe to assume you are dealing with *Infinite-Horizon Discounted Return* as if we had timesteps we would need to sum reward over them (if not explicitly stated that we are doing this then the infinite scenario is a good default choice)


Bellman Equations
- The four Value Functions above all obey *Bellman Equations* -- the fundamental idea behind these equations is this; The value of the Current State is the reward you expect for that State plus the Value of the Next (directly following) State.
- a ~ pi is shorthand for a ~ pi(•|s)
- In order to act optimally, an Agent needs to choose the Action that maximizes value (given the Action-Value estimation function) at each timestep
- Bellman Backup: is a term that comes up in RL, basically the Backup for a State or State-Action pair is the right hand side of the Bellman Equation (i.e the Reward-Plus-Next-Value), see- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations for the Equation to reference

Advantage Functions
- Describing how good an Action is ABSOLUTELY is not always necessary -- we can just determine the best Action from a set of Actions with *Advantage*
- The Relative Advantage of an Action is formalized through the *Advantage Function*, this is basically a function like: A_pi(s, a) -- which boils down to the function returning the expected value for taking Action A in State S over randomly selecting an action (SpinningUp unclear here, could be the Policy recommended Action given the same State or a randomly sampled Action)
- Advantage Functions are crucial to Policy Gradient Methods (worth reading- https://stats.stackexchange.com/questions/434796/policy-gradient-methods-advantages-over-value-based-methods for more information on why Advantage is nice)


MDP Formalism
- Markov Decision Proceses (MDPs) are excellent ways of representing the RL Problem more Formally in a Mathematical sense; the standard formulation for an MDP is a 5-tuple, i.e; [S, A, R, P, p_0] where:
  - S = the set of all valid states
  - A = the set of all valid actions 
  - R = SxAxS --> R (real set of numbers) is the reward function (basically r_t = R(s_t, a_t, s_t+1) for timestep t)
  - P = SxA --> P(S) or the transition probability function with P(s`|s, a) being the probability of transitioning into state s` given state s and action a
  - p_0 = the starting State Distribution (State at timestep 0)
- Markov Property: is a property of MDPs, basically State Transitions only depend on the most recent (last) state and action, earlier history has no effect on the next state transtition (timestep-to-timestep), see the wiki for more info- https://en.wikipedia.org/wiki/Markov_property 


## RL Algorithms Taxonomy, et al.
- Policy Optimization: methods that are On-Policy (see below) and learn directly from the Performance Objective (reward function) -- Policy updates are made iteratively and with the latest version of the Policy (no separate Update Policy idea) -- these methods still learn a Value Approximator (separtate from the Policy) to figure out how to update the Policy. 
- Q-Learning: methods from here learn an approximation of the optimal Action-Value function (Q_theta(s, a) which approximates Q_star(s, a)), these objective functions are often based on Bellman Equations and are typically also Off-Policy (they use past data/Policies to influence current Actions) -- actions here are taken by the argmax of the Action-Value function Q_theta, as that (in the limit) should approximate the optimal Action-Value Function's action
- Interpolations between Policy Optimization & Q-Learning: apparently in some sense these methods are [equivalent](https://arxiv.org/pdf/1704.06440.pdf) -- 

Model Free Vs. Model-Based
- "Modeling an Environment" in RL refers to creating a function that is able to Predict *State Transitions* and *Rewards*. 
- This function can be as simple as providing access to environmental information or it can be a learned function that Predicts the Environment.
- Having a model of the Environment can be useful to an Agent as it opens the door to being able to plan ahead, but complete/correct models of the Environment are not guaranteed and many RL Environments do not have a "ground-truth" model that is show-able or learnable for the Agent. When we can make a model of the environment it can be very useful (as was the case with [AlphaZero](https://arxiv.org/pdf/1712.01815.pdf), which was GIVEN a world-model) and typically leads to better sample efficiency -- but learning a model of the Environment (from experience alone) involves avoiding many local optimas and fail cases (like the Agent maximizing reward of an incorrect world model)
- Algorithms that use these Environment Models are referred to as *Model-Based* methods, and Algorithms that do not are called *Model-Free*. Model-Free Algorithms end up being easier to implement and tune but are not as sample efficient as a good Model-Based Method. 


On-Policy Vs. Off-Policy
- This tradeoff in RL refers to methods that use (off-policy) or do not use (on-policy) old data.
- Although at first counter-intuitive, this terminology makes sense because On-Policy methods are algorithms in which the Behavior Policy (determine next action given current state) and Update Policy are the *same* — whereas in Off-Policy algorithms the Behavior & Update policy are *different*
- This difference is nicely understood via the ways in which these classes of algorithms are implemented:
  - On-Policy methods (think VPG/REINFORCE) calculate the Value of a State-Action pair with the assumption that the Calculated Policy WILL BE FOLLOWED in all future timesteps — the algorithm learns directly from the State-Action pair Values and optimizes its behaviour accordingly (no difference between Behavior Policy and Update Policy) 
    - Because of the direct optimization of the policy, these methods tend to be more Data Hungry (they do not explicitly use old data), but ultimately learn the optimal policy (that is optimizing for maximing Reward through Policy Performance)
    - These trade off Sample Efficiency for Stability (more stable once the Algorithm is thoroughly trained)
	- Implicitly however, On-Policy algorithms techincally do learn from prior data (if modeling policy with a NN say) since the Neural Net's weights are learnt over time from data, but this is a nitpicky point about NN trainings, not necessarily the algorithm or idea behind on-policy as a framework
  - Off-Policy methods (think Q-Learning) calculate the Value of the current State-Action pair but they also have access to an External Policy (the Update Policy) that governs how the Agent should act across Many different State-Action pairs -- i.e. the Value of the current state wrt the Behavior Policy may/can/will be different from the value of the Update Policy. 
    - These also have the Bandits notion of Exploration-vs-Exploitation as the protocol for optimizing a Update Policy is/can be different from the protocol for the Behavior policy. (Basically there is a difference between; State-Action value estimation (Update Policy or the Q-Function) and what is actually used to select the next action in a State-Action pair (Behavior Policy or the Policy))
    - Because of the State-Action value estimation happening w the Q-function, these methods *save what they learn* (i.e. learn from past data) and as a result are more sample efficient than on-policy methods — this is due to their optimizing for Bellman’s Equations (any Q-Function, given enough data (particularly high-reward data) about an Environment, can approximate the most optimal solution/policy for that Environment) — this is great and mathematically ought to always hold but in reality a good solution for Bellman Equations does not mean a good solution for actual reward return (these methods can be brittle wrt estimating the optimal policy, when it works it does so well and with Sample Efficiency but it can also break down, there is no guarantee and performance can be unstable)
- In Summary: On-Policy methods learn directly from the enviroment and optimize for that Reward Signal — Off-Policy methods both estimate Value and Determine what they should do, leading to sample efficient learning at the cost of less optimization towards the True Reward. 


Behavior Vs. Update Policies: a useful way to think about the differences between on and off policy algorithms and to reason about what Policies accomplish for RL Agents
- Behavior Policy: the way in which the Agent will “act” for any given state (likelihood of an action given a state)
- Update Policy: the cumulative estimated return for a given state, if the Agent continues to act in accordance w a specific policy, this is an estimation of the overall reward it expects


Algorithms 
- Vanilla Policy Gradient (VPG)
  - The method for RL before RL was RL (dates back to as early as the 1980s, infamously proposed in Sutton’s 2000 Paper- https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
  - Also commonly referred to as REINFORCE (alternate name)
  - Since this was the original on-policy RL method, it is the precursor to modern on-policy algorithms like PPO and TRPO
- Trust Region Policy Optimization (TRPO)
  - 
- Proximal Policy Optimization (PPO)
  - 
- Deterministic Policy Gradients (DDPG)
  - As foundational to off-policy algorithms as VPG was to on-policy (came to the Universe in 2014, via David Silver- http://proceedings.mlr.press/v32/silver14.pdf)
  - On Arxiv- https://arxiv.org/pdf/1509.02971.pdf (separate paper from Silver's Original above)
  - Closely related to Q-Learning but it also incorporates a deterministic policy (and the two complement each other in training, interpolation between the two traditionally separate methods) 
- Twin Delayed DDPG (TD3)
  - 
- Soft Actor-Critic (SAC)
  - On Arxiv- https://arxiv.org/pdf/1801.01290.pdf 
  - A higher scoring version of DDPG that applies some interesting tricks, namely a Stochastic Policy and Entropy Regularization + more
- Deep Q-Networks (DQN)
  - Originally proposed by the Deepmind folks- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf a foundational method/paper that introduced RL to gaming
  - A variant of the Q-Learning algorithms
- C51
  - On Arxiv- https://arxiv.org/pdf/1707.06887.pdf
  - A method from Q-Learning that learns a Return Distribution, the Expectation fo which is Q_star (learns a probability distribution for the Optimal Q function)
- Model-Predictive Control (MPC)
  - A model-based approach that omits the usage of an Explicit Policy, opting instead to use *Pure Planning* to decide on actions (the Agent computes the full sequence of anticipated Actions everytime it gets an observation from the Environment, and in the next step discards that plan and recomputes)
  - On Wiki- https://en.wikipedia.org/wiki/Model_predictive_control
  - Blog Post for Model-Free Fine-Tuning (MBMF)- https://sites.google.com/view/mbmf (interesting examples of this on Physics Based Control tasks)
- World Models
  - The absolute Mad Lads- David Ha & Jürgen Schmidhuber
  - Custom Page- https://worldmodels.github.io/ (distill publication based)
  - On Arxiv- https://arxiv.org/pdf/1803.10122.pdf


