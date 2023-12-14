# MADDPG

This is a pytorch implementation of MADDPG on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of MADDPG is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- torch=1.1.0

## Quick Start

```shell
$ python main.py
```

Directly run the main.py, then the algrithm will be tested on scenario 'simple_spread' for 10 episodes, using the pretrained model.

## Note

+ We have train agent in "simple_spread",agent can operate according to the distribution I provided.

