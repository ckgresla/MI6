# RL for Wizards

A pocket-repo for RL stuff; Algorithms, pre-trained Agents and Pointers to information


# Info
- "algorithms" contains PyTorch Implementations of each method (ideally importable and well commented, with useful helpers like saving weights)
- "agents" contains saved model weights for Environments 
  - Save New Weights/Controllers as roughly: "<ENV_NAME>-<ALGORITHM>-<STEPs>"
- "housekeeping" has all of the items for... housekeeping (requirements.txt for env replication, etc.)

# Enviroment
running the following is a lightweight way to build the python Environment required to run all of the code in this repo (w Jupyter stuff)
`pip install -r housekeeping/requirements.txt` -- on CPU
`pip install -r housekeeping/requirements-gpu.txt` -- on GPU (check your cuda!)


# Links + Resources
- OpenAI Gym Docs- https://www.gymlibrary.dev/
- SeungEunRho's minimalRL- https://github.com/seungeunrho/minimalRL (great torch implementations of the Classic Stuff)



