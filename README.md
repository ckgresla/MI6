# Agent Focused RL

A pocket-repo for RL stuff; Algorithms, pre-trained Agents, implementations and pointers to information


## Hit-List
Algorithms already implemented;
1. REINFORCE
2. VPG
  -  VPG makes use of Schulman's Generalized Advantage Estimation whereas the REINFORCE implementation uses rewards per timestep (reaction for each action) -- the algorithms are basically the same with the exception of VPG's Value Network/Function, Normalized and Baselined rewards

Algorithms yet to be implemented;
3. TRPO
4. DQN

Tools to leverage
- WarpDrive- https://github.com/salesforce/warp-drive (dare I say "blazingly" fast parallelization for Multi-Agent training via GPU + PyTorch)


## Info
Directory Structure
- "algorithms" contains PyTorch Implementations of each method (has a test use case if file is run @ the command line, intended to be imported thought)
- "agents" contains saved model weights for Environments 
  - Save New Weights/Controllers as roughly: "{ENV_NAME}-{ALGORITHM}-{STEPS}"
- "core" has utilities and code that is shared across Algorithms (environment interaction, etc.)
- "housekeeping" has all of the items for... housekeeping (requirements.txt for env replication, etc.)
- "intel" has some notes and links to helpful resources, papers and information


## Setup
Mainly working with Torch & Gym through Conda install, but see following for full reqs install w pip:
- `pip install -r housekeeping/requirements.txt` -- on CPU
- `pip install -r housekeeping/requirements-gpu.txt` -- on GPU (check your cuda!)


## Links + Resources
Useful stuff to refer to
- OpenAI Gym Docs- https://www.gymlibrary.dev/ (this link somehow rarely comes up if google this)
- SeungEunRho's minimalRL- https://github.com/seungeunrho/minimalRL (great torch implementations)
- garage- https://github.com/rlworkgroup/garage (toolkit for RL research, formerly called "rllab")
- ray- https://github.com/ray-project/ray (RL for Industry, great documentation)
- OpenAI SpinningUp- https://spinningup.openai.com/en/latest/index.html
  - Github- https://github.com/openai/spinningup 

