# Aeroblasters AI

## Project Overview

This project integrates AI into the classic game **Aeroblasters**, allowing an AI agent to control the player's actions, such as moving left, right, and shooting enemies. The goal is to train the AI to maximize its score by destroying enemies, collecting power-ups, and avoiding obstacles. Over time, the AI learns and improves its performance using reinforcement learning techniques.

### Game Authors:
- **Game Author**: Prajjwal Pathak (pyguru)
- **AI Player Author**: Patrick Snyder

### Dates:
- **Game Release Date**: September 30, 2021
- **AI Integration Date**: November 10, 2024

---

## AI Models Implemented

### 1. **Simple Rule-Based AI**

The first version of the AI was a simple **rule-based system** that controlled the player based on predefined conditions:
- **Movement**: The AI moved left or right depending on the position of the closest enemy.
- **Shooting**: The AI shot bullets when an enemy was directly in front of the player.

This approach was easy to implement but lacked adaptability and learning capabilities. The rule-based system could not improve its performance over time and was limited in handling complex situations.

---

### 2. **Deep Q-Learning (DQN) Model**

The next iteration of the AI used a **Deep Q-Network (DQN)**, a popular reinforcement learning algorithm. DQN allows the AI to learn optimal actions by interacting with the environment and receiving rewards for good behavior (e.g., destroying enemies) and penalties for bad behavior (e.g., getting hit by bullets).

#### Key Features of DQN:
- **Q-Learning**: The AI learns a Q-value function that estimates the expected future rewards for taking certain actions in given states.
- **Experience Replay**: The agent stores past experiences in memory and samples them randomly during training to break correlations between consecutive experiences.
- **Epsilon-Greedy Policy**: The AI balances exploration (trying new actions) and exploitation (choosing the best-known action) using an epsilon-greedy policy.

#### DQN Model Architecture:
- A neural network with three fully connected layers:
  - Input: Game state (player position, closest enemy position)
  - Output: Q-values for each possible action (move left, move right, shoot)

#### DQN Performance Logging:
- Performance metrics such as score and total reward were logged after each episode in `dqn_ai_performance.log`.

#### Model Saving:
- The model's weights were saved to `dqn_model.pth` after each episode to allow continued training from previous sessions.

---

### 3. **Proximal Policy Optimization (PPO) Model**

In the latest version of the project, we replaced DQN with **Proximal Policy Optimization (PPO)**, a more advanced reinforcement learning algorithm that is widely used in game environments due to its stability and efficiency.

#### Key Features of PPO:
- **Actor-Critic Architecture**: PPO uses two networks — an actor network that selects actions and a critic network that evaluates those actions.
- **Clipped Objective Function**: PPO prevents large updates to the policy by clipping the objective function, ensuring stable learning.
- **On-Policy Learning**: PPO updates its policy based on data collected from its current policy, making it more stable than off-policy methods like DQN.

#### PPO Model Architecture:
- Actor Network:
  - Input: Game state
  - Output: Action probabilities for each possible action (move left, move right, shoot)

- Critic Network:
  - Input: Game state
  - Output: Estimated value of the current state

#### PPO Performance Logging:
- Performance metrics such as score and total reward are logged after each episode in `ppo_ai_performance.log`.

#### Model Saving:
- The PPO model's weights are saved to `ppo_model.pth` after each episode to allow continued training from previous sessions.

---

## How to Run the Project

### Prerequisites
Before running the game with AI, ensure you have installed all necessary dependencies:

```bash 
pip install pygame torch numpy
```

# Original Game README.md (thanks to pyguru for building a great game to learn from!)
# Aeroblasters

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/check-it-out.svg)](https://forthebadge.com)


Aeroblasters is a simple 2d vertical plane shooter game made using python and pygame. The game can  be run both on pc or android using pydroid3.

Install pydroid3 on Android from here : [pydroid3 playstore](https://play.google.com/store/apps/details?id=ru.iiec.pydroid3&hl=en_IN&gl=US)

<p align='center'>
  <img src='app.png' width=200 height=300>
</p>

## How to Download

Download this project from here [Download Aeroblasters](https://downgit.github.io/#/home?url=https://github.com/pyGuru123/Python-Games/tree/master/Aeroblasters)

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following packages :-
* Pygame

```bash
pip install pygame
```

pygame is already installed in pydroid3, no installation required.

## Usage

Navigate and click main.py to run the game. Then tap on the screen to start playing the game. The objective of the game is to destroy as much enemy planes or choppers as possible without getting destroyed. 

Controls:
* Press left arrow key to go left
* Press right arrow key to go right
* Press space key to go shoot
* Press esc key to quit

When playing with mouse or on pydroid3
* Press left half of game window to go left
* Press right half of game window to go right
* Click on player plane to shoot

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.