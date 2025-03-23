import numpy as np
import random as rd
class RewardManager:
    def __init__(self):
        self.cart_position_weight = 1
        self.termination_penalty = 10.0
        self.alignement_weight = 0.1
        self.upright_weight = 1.0
        self.stability_weight = 0.02  # Weight for the stability reward
        self.old_state = None
        self.length = 0.5  # Pendulum length

    def calculate_reward(self, state, terminated, current_step):
        """
        Calculate the reward based on the current state and termination status.
        
        Args:
            state (tuple): A tuple containing:
                - Cart state: x, x_dot, x_ddot
                - Pendulum states: th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot
            terminated (bool): Whether the episode has terminated due to constraints violation
        
        Returns:
            float: The calculated reward value
        """
        # Extract state components
        x, x_dot, x_ddot = state[0:3]  # Cart state
        th1, th1_dot, th1_ddot = state[3:6]  # First pendulum
        th2, th2_dot, th2_ddot = state[6:9]  # Second pendulum
        th3, th3_dot, th3_ddot = state[9:12]  # Third pendulum
        
        # Calculate pendulum positions (these were previously in the state)
        p1_x = x + self.length * np.sin(th1)
        p1_y = -self.length * np.cos(th1)
        p2_x = p1_x + self.length * np.sin(th2)
        p2_y = p1_y + self.length * np.cos(th2)
        p3_x = p2_x + self.length * np.sin(th3)
        p3_y = p2_y + self.length * np.cos(th3)
        
        self.old_state = state
        
        # x close to 0
        x_penalty = self.cart_position_weight * (abs(x))

        # angles alignement 
        non_alignement_penalty = self.alignement_weight * (abs(th1-th2) + abs(th2-th3) + abs(th3-th1)) / np.pi

        # Uprightness of each node - negate p*_y values because in the physics simulation,
        # negative y means upward (which is what we want to reward)
        # The physics uses a reference frame where positive y is downward
        upright_reward = self.upright_weight * (-p1_y - p2_y - p3_y)

        # Stability reward: penalize high angular velocities and accelerations
        angular_velocity_penalty = (th1_dot**2 + th2_dot**2 + th3_dot**2) / 3.0
        angular_accel_penalty = (th1_ddot**2 + th2_ddot**2 + th3_ddot**2) / 3.0

        stability_penalty = self.stability_weight * (angular_velocity_penalty + angular_accel_penalty)

        # Penalties are negative, rewards are positive
        reward = upright_reward - x_penalty - non_alignement_penalty - stability_penalty
        
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty
            
        return reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty

    def get_reward_components(self, state, current_step):
        reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty = self.calculate_reward(state, False, current_step)
        return {
            'x_penalty': x_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'stability_penalty': stability_penalty,
            'reward': reward
        } 