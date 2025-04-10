import gym
import torch
import pygame
import numpy as np
from tp_env import TriplePendulumEnv
from model import TriplePendulumActor, TriplePendulumCritic
from reward import RewardManager
from metrics import MetricsTracker
import torch.nn.functional as F
import os
import random
from collections import deque
from config import config

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class TriplePendulumTrainer:
    def __init__(self, config):
        self.config = config
        self.reward_manager = RewardManager()
        self.env = TriplePendulumEnv(reward_manager=self.reward_manager, render_mode="human", num_nodes=config['num_nodes'])  # Enable rendering from the start
        
        # Initialize models
        # Ajustement de la dimension d'état en fonction de l'environnement réel
        # Ici, la dimension est basée sur la taille de l'état retourné par reset()
        self.env.reset()
        initial_state = self.env.get_state()
        state_dim = len(initial_state)
        action_dim = 1
        self.actor = TriplePendulumActor(state_dim, action_dim, config['hidden_dim'])
        self.critic = TriplePendulumCritic(state_dim, action_dim, config['hidden_dim'])

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        self.total_steps = 0
        self.max_steps = 500  # Maximum steps per episode
        
        # Exploration parameters
        self.epsilon = 1.0  # Initial random action probability
        self.epsilon_decay = 0.99  # Epsilon decay rate
        self.min_epsilon = 0.001  # Minimum epsilon
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=config['buffer_capacity'])
        
        # Reward normalization
        self.reward_scale = 1.0
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.reward_alpha = 0.001  # For running statistics update
        
        # Create directories for saving results
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Initialize rendering si la méthode existe
        if hasattr(self.env, '_render_init'):
            self.env._render_init()
        
        # Load models
        if config['load_models']:
            self.load_models()
    
    def normalize_reward(self, reward):
        # Update running statistics
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + self.reward_alpha * abs(reward - self.reward_running_mean)
        
        # Avoid division by zero
        std = max(self.reward_running_std, 1e-6)
        
        # Normalize and scale
        normalized_reward = (reward - self.reward_running_mean) / std
        return normalized_reward * self.reward_scale

    def collect_trajectory(self, episode):
        state = self.env.reset()
        state = self.env.get_state()
        done = False
        trajectory = []
        episode_reward = 0
        reward_components = {}
        num_steps = 0
        
        # Réinitialiser le RewardManager
        self.reward_manager.reset()
        
        while not done and num_steps < self.max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:
                action = np.random.uniform(-0.1, 0.1)  # Random action
            else:
                with torch.no_grad():
                    # Concatène les états de la séquence
                    action = self.actor(state_tensor).squeeze().numpy()
                    action /= 5

            # Take step in environment
            next_state, terminated = self.env.step(action)

            # Check for NaN values in state
            if np.isnan(np.sum(next_state)):
                print('state:', next_state)
                raise ValueError("Warning: NaN detected in state")
            
            # Render if rendering is enabled
            if self.env.render_mode == "human":
                rendering_successful = self.env.render(episode, self.epsilon)
                if not rendering_successful:
                    done = True
                    raise ValueError("Warning: Rendering failed")
            
            # Calculate custom reward and components
            custom_reward, _, _, _, _, _, force_terminated = self.reward_manager.calculate_reward(next_state, terminated, num_steps)
            reward_components = self.reward_manager.get_reward_components(next_state, num_steps)
            
            # Store transition in replay buffer with normalized reward
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Convert to tensor

            # Stocker dans le buffer de replay
            self.memory.push(state_tensor, action, custom_reward, next_state_tensor, terminated)
            
            trajectory.append((state_tensor, action, custom_reward, next_state_tensor, terminated))
            episode_reward += custom_reward
            self.total_steps += 1
            num_steps += 1

            if terminated or force_terminated:
                done = True
                break
            
        # Decay exploration parameters
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"Episode {episode} ended with {num_steps} steps")

        return trajectory, episode_reward, reward_components
    
    def update_networks(self):
        if len(self.memory) < self.config['batch_size']:
            return {"critic_loss": 0, "actor_loss": 0}  # Not enough samples yet

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Reshape states to handle sequence properly
        batch_size = states.shape[0]
        
        # Reshape states from [batch_size, seq_length, state_dim] to [batch_size, seq_length*state_dim]
        states_reshaped = torch.cat([states[i] for i in range(batch_size)], dim=0).view(batch_size, -1)
        next_states_reshaped = torch.cat([next_states[i] for i in range(batch_size)], dim=0).view(batch_size, -1)
        
        # Update critic
        current_q = self.critic(states_reshaped, actions)
        with torch.no_grad():
            next_actions = self.actor(next_states_reshaped)
            target_q = rewards + (1 - dones) * self.config['gamma'] * self.critic(next_states_reshaped, next_actions)
        
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Add gradient clipping for critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update actor - we want to maximize the critic output (Q-value)
        # Since optimizers perform minimization by default, we use negative sign to turn maximization into minimization
        # This is correct for DDPG: we want the actor to choose actions that maximize Q-values
        actor_loss = -self.critic(states_reshaped, self.actor(states_reshaped)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Add gradient clipping for actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def train(self):
        for episode in range(self.config['num_episodes']):
            print(f"Episode {episode} started")
            
            # Activer ou désactiver le rendu en fonction du numéro d'épisode
            if episode % 50 == 0 and episode > 200:
                self.env.render_mode = "human"
            else:
                self.env.render_mode = None
                
            # Collect trajectory and store in replay buffer
            trajectory, episode_reward, reward_components = self.collect_trajectory(episode)
            
            # Only update after we have enough samples
            losses = {"critic_loss": 0, "actor_loss": 0}
            if len(self.memory) >= self.config['batch_size']:
                # Perform multiple updates per episode
                for _ in range(self.config['updates_per_episode']):
                    update_losses = self.update_networks()
                    # Accumulate losses for reporting
                    losses['critic_loss'] += update_losses['critic_loss']
                    losses['actor_loss'] += update_losses['actor_loss']
                
                # Average the losses
                losses['critic_loss'] /= self.config['updates_per_episode']
                losses['actor_loss'] /= self.config['updates_per_episode']
            
            # Track metrics
            self.metrics.add_metric('episode_reward', episode_reward)
            self.metrics.add_metric('actor_loss', losses['actor_loss'])
            self.metrics.add_metric('critic_loss', losses['critic_loss'])
            
            # Track reward components
            for component_name, value in reward_components.items():
                self.metrics.add_metric(component_name, value)
            
            # Periodic display and saving
            if episode % 100 == 99:
                avg_reward = self.metrics.get_moving_average('episode_reward')
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
                
                # Save graphs
                self.metrics.plot_metrics(f'results/metrics.png')
                
                # Save model
                self.save_models(f"models/checkpoint")
    
    def save_models(self, path):
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critic.state_dict(), path + '_critic.pth')
        torch.save(self.actor_optimizer.state_dict(), path + '_actor_optimizer.pth')
        torch.save(self.critic_optimizer.state_dict(), path + '_critic_optimizer.pth')

    def load_models(self):
        if os.path.exists('models/checkpoint_actor.pth'):
            print("Loading models")
            self.actor.load_state_dict(torch.load('models/checkpoint_actor.pth', weights_only=True))
            self.critic.load_state_dict(torch.load('models/checkpoint_critic.pth', weights_only=True))
            self.actor_optimizer.load_state_dict(torch.load('models/checkpoint_actor_optimizer.pth', weights_only=True))
            self.critic_optimizer.load_state_dict(torch.load('models/checkpoint_critic_optimizer.pth', weights_only=True))

if __name__ == "__main__":    
    trainer = TriplePendulumTrainer(config)
    trainer.train() 