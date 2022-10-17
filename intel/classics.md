# Classic RL Papers
Some of the "timeless" papers in rl


## General Info
- 


## Papers



### Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE), Williams (1992)
- Link- https://link.springer.com/content/pdf/10.1007/BF00992696.pdf 
- See `policy_gradients.md` for the notes


### Reinforcement Learning for Continuous Action using Stochastic Gradient Ascent, Kimura and Kobayashi, 1998.
Link- http://sysplan.nams.kyushu-u.ac.jp/gen/papers/ias98.pdf
- Main Contributions: Pole Balancing task w Policy Optimization


### PEGASUS: A Policy search Method for Large MDPs and POMDPs, Ng & Jordan, 2000.
- Link- https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Ng+Jordan:2000.pdf
- Main Contributions: the PEGASUS Method?


### Reinforcement Learning of Motor Skills with Policy Gradients, Peters and Schaal, 2008. 
- Link- https://is.mpg.de/fileadmin/user_upload/files/publications/Neural-Netw-2008-21-682_4867%5B0%5D.pdf
- Main Contributions: teach a robot arm to hit a baseball and a review of Policy Optimization SOTA (as of 2008)
- At the time, folks were using things like "spline nodes" to create controllers for robot/locomotion tasks, these things were pretty brittle and did not "understand" or were not robust within their environments
  - These Splines were used to control "Motor Primitives", see- https://www.sciencedirect.com/science/article/abs/pii/S095943881500077X
  - A Motor Primitive is a "building block" -- a series of motor primitives can produce complex actions/movements through manipulation (think of these as lego blocks for movements) the concepts for these come from Neuroscience and Biology but seem to also have some weight in the Robotics space
    - In Robotics, there is an additional requirement on RL, any changes to the Policy **CANNOT** be drastic (as quick changes can damage the robot or the environment)
- Supervised Learning (think of things like expert systems or imitation learning as opposed to Digit Classification, for what the Author is trying to talk about here) is great but is not as generally applicable as we would need to *train* Agents to do everything (as opposed to *learning* everything) -- furthermore the environments that Agents will operate in will ALWAYS have features/mechanisms that go beyond the training data (Gödel strikes yet again)
  - Supervised methods then are not sufficient for training "true" Agents (things that can swing a tennis racket to hit a ball, play drums, walk on wild terrain, etc.)
- Also at this time RL was limited in the space of "high-dimensional continuous control style" tasks (Schulman was at prom or something), the SOTA was limited to 4 degrees of freedom and non-able to work with parameterized policies (Neural Networks in the Mix) -- with the exception of Policy Optimization methods (Sutton laughs)
- Trajectories (sequences of States and Actions, i.e; [s1, a1, s2, a2, ...]) are commonly referred to as "history" or "roll-out" in the literature
  - Instead of N or t to denote the number of these values, H can also be used (referring to the "Horizon" of the Trajectory)
- In Reality, Robots/Agents do not have very comprehensive "Observations" of their environments, therefore any Policy Optimization Algorithm will have to be able to estimate the *actual* policy gradient from data generated via the execution of a task (i.e, what did it feel like to do X and given that "internal" observation, what would it be like to do X *better*)
  - Not Explicitly mapping the world here, still true Policy Optimization (model-free), bias is not baked-in
  - Good way of thinking about how Rewards are actually accessed (from an Agent's perspective) -- they only percieve a limited slice of the actual environment dynamics and get some small notion of reward based on actions they take, that is a non-clear way of trying to get an Agent to do something...

- Finite-Difference Methods: are pretty old-school (dating back to the 1950s) and basically involve creating estimates of the expected return via varying the policy parameters slightly (testing over several sets of slightly different policy parameters), this method can lead to some problems as it requires an understanding of what changing policy parameters might do (if you jack up the torque for the knee motor primitive you'll start falling, etc.) --> this method really requires Human Supervision to get correct (think tuning the Hyperparameters of a small NN by hand to see what happens, then do it for trying to get some hand engineered system to do locomotion, not a good time)
- Likelihood Ratios & REINFORCE: the coolest thing about this method is that we do not need to compute derivatives wrt the underlying control parameters (i.e; generating distribution for *optimal* control), rather by running a Policy in an environment and then estimating the reward for running that policy, we can compute the derivative of the Rewards wrt to the Network Parameters (and train towards optimal without knowing what explicitly is optimal, just follow the rewards), still a bit of a non-clear reward signal but it works!
  - The variance of Policy Estimates in REINFORCE can be reduced with the addition of a baseline (like GAE) 

**Vanilla Policy Gradient Approaches**
- Variance is the largest culprit of these Algorithms (I can personally attest to this, VPG can do some whacky stuff)
  - If we fail to include a baseline, things break quickly (error can grow cubically in fact)
  - Including a baseline does a great bit to curtail the variance
  - Things like excluding future actions (only updating the Policy based on the rewards up to point t) also helps reduce the variance (this is where the sub-methods of REINFORCE, Sutton's Policy Gradient Theorem (PGT) and Baxter & Bartlett's G(PO)MDP) first popped up)
- Optimal Baselines reduce the Variance of each element in the Gradient *without* biasing the Gradient Estimate
- Estimating the Action-Value function (as listed here) is analagous to trying to fit a parabola with a linear line, not a good time (a potential solution for this is then proposed with Actor-Critic Methods)
  - In VPG (with GAE) the Action-Value function is ingeniously created (at least how we have implemented it here) -- the Q(s, a) function is approximated (or calculated) by taking the actual reward recieved at timestep t and the estimated reward (output of the Value function, i.e; V(state) --> estimated_reward) for this timestep (ve_t) and the *next* timestep (ve_t+1). These terms come together as follows: 
    - delta = r_t + (gamma * ve_t+1) - ve_t --> which basically approximates the *value* for an action in a state by subtracting the value of the current state (could brush up on)

**Natural Actor-Critic**
- Policy Gradients with Baselines, still kind of suck at learning (sample inefficient + run into seemingly every possible wall)
- One solution is the **Natural Policy Gradient** (NPG) (great visualization of the difference in learning spaces for the VPG and NPG in Figure 1), this method was first introduced by Kakade (2002) as the "average natural policy gradient" (and later was validated as the "True Natural Gradient") -- here are some of the features that make this method really nice: 
  - Convergence to local optimums are guaranteed
  - NPG takes a more direct path towards the Gradient (as opposed to VPG), meaning it converges more quickly and also does not prematurely converge
  - The NPG is also *covariant*, that is the orientation of the "coordinate frame" for expressing the policy parameters (initialization of NN) is independent of the optimization towards the gradient
  - NPG also averages out the stochastic Policy's influence (analytically or in the equation) so it needs less data points for the gradient estimate (whereas VPG relies on sampling)
- Reward Gradient Estimation is the heart of the Policy Gradient problem, NPG gets at achieving *covaraint gradient descent* which basically runs Gradient Descent on an *invariant* distance measure, one that points towards optimal learning from anywhere in the optimization landscape 
- This paper then introduces the **Natural Actor-Critic** (the next interesting step after the introduction of "Actor-Critic methods" in Barto, Sutton & Anderson, 1983)
  - The Critic (is not the one who counts!): evaluates a current policy (pi) and provides an approximation of the Value Function, in this work it is formulated as a regression problem (and can be approximated with a NN)
  - The Actor: is our policy network from VPG, a NN that takes in an observation and outputs action logits
- Findings: the visualizations in figures 2, 3 and especially 4 showcase the differences in learning for each of the 5 algorithms; [finite-difference, VPG w Constant Baseline, VPG w Time-variant Baseline, Episodic Natural Actor-critic w single offset basis function, Episodic Natural Actor-critic w time-variant offset basis function]
  - VPG and the Natural Actor-Critic methods achieve the same "ultimate" performance, it just requires a few more orders of magnitude in rollouts (training data) for the VPG methods to catch up (if they are able to)
  - Finite-Difference Gradient did not perform all that well and was trapped in local minima for effectively all of the tested tasks (no stochastic policy so no real exploration was done, this is a criticism of the algorithm that goes back to its inception)


