# Aeroblasters AI

# PPO based reinforcement learning model training environment that plays Aeroblasters
# AI Player Author: Patrick Snyder
# Date: Sunday, 10 November, 2024

# Game Author : Prajjwal Pathak (pyguru)
# Date : Thursday, 30 September, 2021

import os
import random
import pygame
from objects import Background, Player, Enemy, Bullet, Explosion, Fuel, Powerup, Button, Message, BlinkingText
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging

pygame.init()

# Screen settings
SCREEN = WIDTH, HEIGHT = 288, 512

info = pygame.display.Info()
width = info.current_w
height = info.current_h

if width >= height:
    #win = pygame.display.set_mode(SCREEN, pygame.NOFRAME)
    win = pygame.display.set_mode(SCREEN,pygame.SCALED)
else:
	#win = pygame.display.set_mode(SCREEN, pygame.NOFRAME | pygame.SCALED | pygame.FULLSCREEN)
    win = pygame.display.set_mode(SCREEN, pygame.SCALED | pygame.FULLSCREEN)

pygame.display.set_caption('Proximal Policy Optimization')
clock = pygame.time.Clock()
FPS = 60

# LOGGING *********************************************************************
# Configure logging
logging.basicConfig(filename='ppo_ai_performance.log', level=logging.INFO,
                    format='%(asctime)s, %(message)s')

def log_performance(score, reward, episode):
    """Log AI performance after each episode."""
    logging.info(f'Episode: {episode}, Score: {score}, Reward: {reward}')
                 
# AI CONTROL CONFIG ***********************************************************
model_path = "ppo_model.pth" # Start from a pre-trained model
state_size = 4  # State vector: [player_x_position, player_y_position, closest_enemy_x_position, closest_enemy_y_position]
action_size = 3  # Action space: [move_left, move_right, shoot]
batch_size = 32  # Size of minibatch for training

# COLORS **********************************************************************

WHITE = (255, 255, 255)
BLUE = (30, 144,255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 20)

# IMAGES **********************************************************************

plane_img = pygame.image.load('Assets/plane.png')
logo_img = pygame.image.load('Assets/logo.png')
fighter_img = pygame.image.load('Assets/fighter.png')
clouds_img = pygame.image.load('Assets/clouds.png')
clouds_img = pygame.transform.scale(clouds_img, (WIDTH, 350))

home_img = pygame.image.load('Assets/Buttons/homeBtn.png')
replay_img = pygame.image.load('Assets/Buttons/replay.png')
sound_off_img = pygame.image.load("Assets/Buttons/soundOffBtn.png")
sound_on_img = pygame.image.load("Assets/Buttons/soundOnBtn.png")


# BUTTONS *********************************************************************

home_btn = Button(home_img, (24, 24), WIDTH // 4 - 18, HEIGHT//2 + 120)
replay_btn = Button(replay_img, (36,36), WIDTH // 2  - 18, HEIGHT//2 + 115)
sound_btn = Button(sound_on_img, (24, 24), WIDTH - WIDTH // 4 - 18, HEIGHT//2 + 120)


# FONTS ***********************************************************************

game_over_font = 'Fonts/ghostclan.ttf'
tap_to_play_font = 'Fonts/BubblegumSans-Regular.ttf'
score_font = 'Fonts/DalelandsUncialBold-82zA.ttf'
final_score_font = 'Fonts/DroneflyRegular-K78LA.ttf'

game_over_msg = Message(WIDTH//2, 230, 30, 'Game Over', game_over_font, WHITE, win)
score_msg = Message(WIDTH-70, 28, 30, '0', final_score_font, RED, win)
final_score_msg = Message(WIDTH//2, 280, 30, '0', final_score_font, RED, win)
tap_to_play_msg = tap_to_play = BlinkingText(WIDTH//2, HEIGHT-60, 25, "Tap To Play",
				 tap_to_play_font, WHITE, win)
reward_msg = Message(WIDTH-70, 58, 30, '0', final_score_font, GREEN, win)
episode_msg = Message(WIDTH-70, 88, 30, '1', final_score_font, BLUE, win)

# GROUPS & OBJECTS ************************************************************

bg = Background(win)
p = Player(144, HEIGHT - 100)

enemy_group = pygame.sprite.Group()
player_bullet_group = pygame.sprite.Group()
enemy_bullet_group = pygame.sprite.Group()
explosion_group = pygame.sprite.Group()
fuel_group = pygame.sprite.Group()
powerup_group = pygame.sprite.Group()

# FUNCTIONS *******************************************************************

def shoot_bullet():
	x, y = p.rect.center[0], p.rect.y

	if p.powerup > 0:
		# Shoot multiple bullets if powerup is active
		for dx in range(-3, 4):
			b = Bullet(x, y, 4, dx)
			player_bullet_group.add(b)
		p.powerup -= 1
	else:
		# Regular shooting with two bullets
		b = Bullet(x-30, y, 6)
		player_bullet_group.add(b)
		b = Bullet(x+30, y, 6)
		player_bullet_group.add(b)

def reset():
	# Reset game state when restarting.
	enemy_group.empty()
	player_bullet_group.empty()
	enemy_bullet_group.empty()
	explosion_group.empty()
	fuel_group.empty()
	powerup_group.empty()

	p.reset(p.x, p.y)

	global moving_left, moving_right, score, reward
	moving_left = False
	moving_right = False
	score = 0
	reward = 0
# AI CONTROL ******************************************************************
# PPO Neural Network Model for Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

# PPO Agent Class
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # discount factor
        self.eps_clip = 0.2  # clipping range for PPO
        self.learning_rate = 0.001
        
        # Initialize actor-critic model
        self.model = ActorCritic(state_size, action_size)
        
        # Optimizer for both actor and critic networks
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function for critic (value function)
        self.value_loss_fn = nn.MSELoss()

    def act(self, state):
        """Select an action based on the current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        
        # Sample an action from the probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), action_dist.log_prob(action)

    def compute_advantage(self, rewards, values):
        """Compute advantages using discounted rewards."""
        returns = []
        discounted_sum = 0
        
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).detach()
        
        advantages = returns - torch.FloatTensor(values).detach()
        
        return advantages

    def update(self, states, actions, log_probs_old, rewards, values):
        """Update policy and value networks using PPO loss."""
        
        advantages = self.compute_advantage(rewards, values)

        for _ in range(4):  # Multiple epochs of optimization
            for i in range(len(states)):
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)
                _, state_value = self.model(state_tensor)
                
                # Get new log probability of actions under current policy
                new_action_probs, _ = self.model(state_tensor)
                new_action_dist = torch.distributions.Categorical(new_action_probs)
                new_log_prob = new_action_dist.log_prob(torch.tensor(actions[i]))
                
                # Compute ratio of new and old probabilities
                ratio = torch.exp(new_log_prob - log_probs_old[i])
                
                # Compute surrogate loss with clipping
                surr1 = ratio * advantages[i]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                actor_loss = -torch.min(surr1, surr2)

                # Critic loss (value function loss)
                critic_loss = self.value_loss_fn(state_value.squeeze(), torch.tensor(rewards[i]))

                # Total loss is sum of actor and critic losses
                loss = actor_loss + 0.5 * critic_loss
                
                # Backpropagation and optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def load(self, name):
        """Load model weights."""
        self.model.load_state_dict(torch.load(name,weights_only=True))

    def save(self, name):
        """Save model weights."""
        torch.save(self.model.state_dict(), name)
        

def ai_control():
    # AI control logic for moving and shooting.
    global moving_left, moving_right

    # Find closest enemy
    closest_enemy = None
    min_distance = float('inf')

    # Iterate over all enemies in enemy_group to find the closest one
    for enemy in enemy_group:
        distance = abs(enemy.rect.x - p.rect.x)  # Calculate horizontal distance
        if distance < min_distance:
            min_distance = distance
            closest_enemy = enemy

    # Get current game state
    state = np.array([p.rect.x / WIDTH,
                        p.rect.y / HEIGHT,
                        closest_enemy.rect.x / WIDTH if closest_enemy else 0,
                        closest_enemy.rect.y / HEIGHT if closest_enemy else 0])

    # Get action from agent (PPO policy)
    action, log_prob_action = agent.act(state)

    # Map action to game controls (e.g., 0: move left, 1: move right, 2: shoot)
    if action == 0:
        moving_left = True
        moving_right = False
    elif action == 1:
        moving_left = False
        moving_right = True
    elif action == 2:
        shoot_bullet()

    return state, action, log_prob_action

# VARIABLES *******************************************************************

level = 1
plane_destroy_count = 0
plane_frequency = 5000
start_time = pygame.time.get_ticks()

moving_left = False
moving_right = False

home_page = False
game_page = True
score_page = False

score = 0
sound_on = False

running = True
done = False

# Main Game Loop **************************************************************
# Create PPO AI Agent
agent = PPOAgent(state_size=state_size, action_size=action_size)

# Check if pre-trained model exists and load it
if os.path.exists(model_path):
    logging.info("Loading pre-trained model...")
    agent.load(model_path)
else:
    logging.info("No pre-trained model found. Starting from scratch.")
episode = 1
reward = 0

while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False

    if game_page:
        state, action, log_prob_action = ai_control()   # Use updated ai_control()

        current_time=pygame.time.get_ticks()
        delta_time=current_time - start_time

        if delta_time >= plane_frequency:
            if level == 1:
                type = 1
            elif level == 2:
                type = 2
            elif level == 3:
                type = 3
            elif level == 4:
                type = random.randint(4, 5)
            elif level == 5:
                type = random.randint(1, 5)

            x = random.randint(10, WIDTH - 100)
            e = Enemy(x, -150, type)
            enemy_group.add(e)
        
            start_time = current_time

        if plane_destroy_count:
            if plane_destroy_count % 5 == 0 and level < 5:
                level += 1
                # Reward for moving up a level
                reward += 20
                plane_destroy_count = 0

        # Update fuel levels as we progress
        p.fuel -= 0.05

        bg.update(1)
        win.blit(clouds_img, (0, 70))

        p.update(moving_left, moving_right, explosion_group)
        p.draw(win)

        player_bullet_group.update()
        player_bullet_group.draw(win)
        enemy_bullet_group.update()
        enemy_bullet_group.draw(win)
        explosion_group.update()
        explosion_group.draw(win)
        fuel_group.update()
        fuel_group.draw(win)
        powerup_group.update()
        powerup_group.draw(win)
        enemy_group.update(enemy_bullet_group, explosion_group)
        enemy_group.draw(win)
        
        if p.alive:
            player_hit = pygame.sprite.spritecollide(p, enemy_bullet_group, False)
            for bullet in player_hit:
                p.health -= bullet.damage
                # Punishment for being hit by a bullet
                reward -= 10
                x, y = bullet.rect.center
                explosion = Explosion(x, y, 1)
                explosion_group.add(explosion)

                bullet.kill()

            for bullet in player_bullet_group:
                planes_hit = pygame.sprite.spritecollide(bullet, enemy_group, False)
                for plane in planes_hit:
                    plane.health -= bullet.damage
                    
                    # Reward for hitting a plane
                    reward += 1
        
                    if plane.health <= 0:
                        x, y = plane.rect.center
                        rand = random.random()
                        if rand >= 0.9:
                            power = Powerup(x, y)
                            powerup_group.add(power)
                        elif rand >= 0.3:
                            fuel = Fuel(x, y)
                            fuel_group.add(fuel)

                        plane_destroy_count += 1
        
                        # Reward for destroying a plane
                        reward += 10

                    x, y = bullet.rect.center
                    explosion = Explosion(x, y, 1)
                    explosion_group.add(explosion)

                    bullet.kill()

            player_collide = pygame.sprite.spritecollide(p, enemy_group, True)
            if player_collide:
                x, y = p.rect.center
                explosion = Explosion(x, y, 2)
                explosion_group.add(explosion)

                x, y = player_collide[0].rect.center
                explosion = Explosion(x, y, 2)
                explosion_group.add(explosion)
                
                p.health = 0
                p.alive = False

                # Punishment for colliding with a plane
                reward -= 20

            if pygame.sprite.spritecollide(p, fuel_group, True):
                p.fuel += 25
                # Reward for picking up fuel
                reward += 5
                if p.fuel >= 100:
                    p.fuel = 100

            if pygame.sprite.spritecollide(p, powerup_group, True):
                p.powerup += 2
                # Reward for picking up power-up
                reward += 5

        if not p.alive or p.fuel <= -10:
            if len(explosion_group) == 0:
                # Log performance and save model after training session ends.
                log_performance(score,reward,episode)
                agent.save("ppo_model.pth")
                episode += 1
                reward = 0
                game_page = True
                done = True
                reset()

        score += 1
        # Reward for increasing score
        #reward += 1
        score_msg.update(f'S: {score}')
        reward_msg.update(f'R: {reward}')

        fuel_color = RED if p.fuel <= 40 else GREEN
        pygame.draw.rect(win, fuel_color, (30, 20, p.fuel, 10), border_radius=4)
        pygame.draw.rect(win, WHITE, (30, 20, 100, 10), 2, border_radius=4)
        pygame.draw.rect(win, BLUE, (30, 32, p.health, 10), border_radius=4)
        pygame.draw.rect(win, WHITE, (30, 32, 100, 10), 2, border_radius=4)
        win.blit(plane_img, (10, 15))

    episode_msg.update(f'E: {episode}')
    pygame.draw.rect(win, WHITE, (0,0, WIDTH, HEIGHT), 5, border_radius=4)
    clock.tick(FPS)
    pygame.display.update()

pygame.quit()