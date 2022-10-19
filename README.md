# Agent Focused RL

A pocket-repo for RL stuff; Algorithms, pre-trained Agents, implementations and pointers to information


## Hit-List
Implemented: 
- 

Algorithms QUESTIONABLY implemented;
1. REINFORCE
2. VPG
  -  VPG makes use of Schulman's Generalized Advantage Estimation whereas the REINFORCE implementation uses rewards per timestep (reaction for each action) -- the algorithms are basically the same with the exception of VPG's Value Network/Function, Normalized and Baselined rewards


## Info
Directory Structure
- "agents" contains saved model weights for Environments 
  - Save New Weights/Controllers as roughly: "{ENV_NAME}-{ALGORITHM}-{STEPS}"
- "housekeeping" has all of the items for... housekeeping (requirements.txt for env replication, etc.)
- "intel" has some of my notes and links to; helpful resources, papers and information
- "mi6" holds the code for all the magic
  - "algorithms" contains PyTorch Implementations of each method (runnable at terminal or importable)
  - "core" has utilities and code that is sensible to share across Algorithms (environment interaction, etc.)


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

Tools to leverage
- WarpDrive- https://github.com/salesforce/warp-drive (dare I say "blazingly" fast parallelization for Multi-Agent training via GPU + PyTorch)
- PeaRL- https://github.com/LondonNode/Pearl (a set of python/torch based modules for crafting Deep RL agents)


