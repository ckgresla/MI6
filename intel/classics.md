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


### Reinforcement Learning of Motor Skills with Policy Gradients, Peters and Schaal, 2008. 
- Link- https://is.mpg.de/fileadmin/user_upload/files/publications/Neural-Netw-2008-21-682_4867%5B0%5D.pdf
- Main Contributions: teach a robot arm to hit a baseball and a review of Policy Optimization SOTA (as of 2008)
- At the time, folks were using things like "spline nodes" to create controllers for robot/locomotion tasks, these things were pretty brittle and did not "understand" or were not robust within their environments
  - These Splines were used to control "Motor Primitives", see- https://www.sciencedirect.com/science/article/abs/pii/S095943881500077X
- Supervised Learning (think of things like expert systems or imitation learning as opposed to Digit Classification) is great but is not as generally applicable as we would need to train Agents to do everything -- furthermore the environments that Agents will operate in will ALWAYS have features/mechanisms that go beyond the training data (Gödel strikes yet again)
  - Supervised methods then are not sufficient for training "true" Agents (things that can swing a tennis racket to hit a ball, play drums, walk on wild terrain, etc.)
- Also at this time RL was limited in the "high-dimensional continuous control style tasks" (Schulman was at prom or something), the SOTA was limited to 4 degrees of freedom and non-able to work with parameterized policies (Neural Networks in the Mix) -- with the exception of Policy Optimization methods (Sutton laughs in genius)
- Trajectories (sequences of States and Actions, i.e; [s1, a1, s2, a2, ...]) are commonly referred to as "history" or "roll-out" in the literature
  - Instead of N or t to denote the number of these values, H can also be used (referring to the "Horizon" of the Trajectory)
- **left off at page 3, start of section 2** 


