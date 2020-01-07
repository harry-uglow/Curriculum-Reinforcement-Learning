Main development repository for MEng in Computing (Artificial Intelligence) final project titled "CuRL: Curriculum Reinforcement Learning for Goal-Oriented Robot Control" (originally "Deep Reinforcement Learning in Simulation with Real-world Fine Tuning"). Project aims to develop a pipeline for learning robotic control tasks by first training in simulation before transferring to a real robot.

# Deep-RL-Sim2Real

This project uses PPO as its main RL algorithm. [ikostrikov's implementation](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) was used as a starting point for this repository. ikostrikov_license.txt contains the license for this implementation. The project was forked at commit ddb89a5cc4df36396d17a73d4b6631fa3caca3b4 - any changes since then are my own.

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# Pytorch and other requirements
pip install -r requirements.txt

# Baselines (with some personal edits)
git clone https://github.com/harry-uglow/baselines.git
cd baselines
pip install -e .
```

Install V-REP, move remoteApi.so (or .dylib) to root folder.

## Branches

* master - most code is here in Python 3.7
* python2_... - branches with this prefix are written in Python 2.7 because \
required packages for connecting to Sawyer robot are Python 2 ONLY
