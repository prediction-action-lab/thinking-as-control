## Overview

This repository contains code for the paper "When can model-free reinforcement learning be enough for thinking?" that appeared at NeurIPS 2025.
The code can be used to reproduce the Gridworld thought MDP experimental results and reproduce the policy iteration example given in Section 4.

## Gridworld Thought MDP

### Pre-training

To pre-train a model, use `pre-training.py` as follows:

```
python pre-training.py --model_save_path=<path/to/save/model.pth>
```
The goal of pre-training is to prime the model so that thinking actions will trigger the agent to be more likely to take certain follow-on actions. You can test if this priming was successful by using the same script as follows:
```
python pre-training.py --model_load_path=<path/to/load/model.pth> --prompt_thinking --mask_thinking
```
With these options, a pre-trained model will be loaded and evaluated in the two stage Gridworld used in the RL experiments; pre-training will not be ran again. 
The above command will force the model to think 'A' on the first time-step and 'B' when the agent reaches the bottom right corner. Thinking actions are masked out on other time-steps. If pre-training successfully primed the model to learn to think then the success rate on the task will be significantly higher compared to running without these two options.

### Reinforcement Learning

To reproduce the RL portion of the experiments, the command is:
```
python gridworld-rl.py --model_path=</path/to/pretrained/model.pth> --output_file=</path/to/results/file.npy> --seed=42
```
This command will load a pre-trained model checkpoint from the provided path, run RL from that checkpoint, and save the success rate per iteration to the provided results file path.
To run the "No-Think" baseline, use the option `--mask_thinking`. To run the "Scratch-*" baselines, simply omit the model_path.

### Plotting Results

Coming soon...
