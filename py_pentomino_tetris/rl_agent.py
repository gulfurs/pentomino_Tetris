"""
Reinforcement Learning Agent for Pentomino Tetris
Using a simple linear function approximator instead of Deep Q-Learning
"""

import numpy as np
import random
from collections import deque
from constants import GRID_WIDTH, GRID_HEIGHT


class PentominoGameState:
    """A class to convert the game state to a format suitable for the RL agent"""
    def __init__(self, game):
        self.game = game
        
    def get_state_matrix(self):
        """
        Convert the game state to a binary matrix representation
        
        Returns:
            np.array: A binary matrix representing the game state
        """
        # Create a matrix representing the current board state
        board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)
        
        # Mark filled cells as 1
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game.grid[y][x] is not None:
                    board[y, x] = 1
        
        # Mark current piece position as 2
        for x, y in self.game.current_piece.get_coords():
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                board[y, x] = 2
                
        return board
    
    def get_state_features(self):
        """
        Extract relevant features from the game state
        
        Returns:
            np.array: A feature vector
        """
        board = self.get_state_matrix()
        
        # Column heights
        column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                column_heights.append(0)
        
        # Number of holes (empty cells with filled cells above them)
        holes = 0
        for x in range(GRID_WIDTH):
            found_block = False
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    found_block = True
                elif found_block:
                    holes += 1
        
        # Height differences between adjacent columns
        height_diffs = [abs(column_heights[i] - column_heights[i-1]) for i in range(1, GRID_WIDTH)]
        
        # Count complete and near-complete lines
        complete_lines = 0
        near_complete_lines = 0  # Lines that are almost complete (good targets)
        for y in range(GRID_HEIGHT):
            filled_count = sum(1 for x in range(GRID_WIDTH) if board[y, x] > 0)
            if filled_count == GRID_WIDTH:
                complete_lines += 1
            elif filled_count >= GRID_WIDTH - 2:  # Only 1-2 cells missing
                near_complete_lines += 1
        
        # Bumpiness - sum of differences between adjacent columns
        bumpiness = sum(height_diffs)
        
        # Max height and average height
        max_height = max(column_heights) if column_heights else 0
        avg_height = sum(column_heights) / GRID_WIDTH if column_heights else 0
        
        # Line opportunity - how many lines could be completed with good placements
        line_opportunity = near_complete_lines
        
        # Current piece position - how centered is the current piece?
        current_piece_x = self.game.current_piece.x
        center_distance = abs(current_piece_x - GRID_WIDTH // 2) / GRID_WIDTH  # Normalized distance from center
        
        # Aggregate features - now with more useful information
        features = np.array([
            holes,                    # Holes (bad)
            bumpiness * 0.5,          # Bumpiness (bad, but scaled down)
            max_height / GRID_HEIGHT, # Max height, normalized (high is bad)
            complete_lines * 2,       # Complete lines (good, weighted more)
            near_complete_lines,      # Near-complete lines (good)
            line_opportunity,         # Line opportunity (good)
            avg_height / GRID_HEIGHT, # Average height, normalized (high is bad)
            center_distance,          # Distance from center (bad)
            1.0 if self.game.current_piece.shape_name == 'I' else 0.0,  # I piece (good for clearing)
            1.0 if self.game.current_piece.shape_name in ['T', 'X'] else 0.0,  # Versatile pieces
        ])
        
        return features


class LinearAgent:
    """
    Simplified Linear Q-learning agent:
    - Immediate online updates (one-step Q-learning)
    - True epsilon-greedy with external exponential decay
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Initialize weights
        self.weights = np.random.uniform(-0.05, 0.05, (action_size, state_size))
        
        # Initialize weights with bias toward good Tetris strategy
        # Penalize holes and height heavily
        self.weights[:, 0] *= -2.0  # Holes are very bad
        self.weights[:, 2] *= -1.5  # Height is bad
        self.weights[:, 4] *= 1.5   # Near-complete lines are good
        
        # Bias left/right actions positively to encourage horizontal movement
        self.weights[0, :] += 0.2  # Strong bias for left 
        self.weights[1, :] += 0.2  # Strong bias for right
        self.weights[4, :] -= 0.5  # Very strong negative bias for hard drops

        # Learning parameters
        self.alpha = 0.01
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
    
    def predict(self, state):
        """Return Q-values for all actions"""
        return np.dot(self.weights, state[0])

    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.predict(state)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        """One-step Q-learning weight update with emphasis on positive rewards"""
        # Skip update if any features contain NaN/inf
        if not (np.isfinite(state[0]).all() and np.isfinite(next_state[0]).all()):
            return
            
        q_current = np.dot(self.weights[action], state[0])
        if not np.isfinite(q_current):
            return
            
        q_next = 0 if done else np.max(self.predict(next_state))
        if not np.isfinite(q_next):
            return
            
        target = reward + self.gamma * q_next
        error = target - q_current
        
        if not np.isfinite(error):
            return
        
        # Use different learning rates based on reward
        # Line clears get much higher learning rate
        learning_rate = self.alpha * 5 if reward > 50 else self.alpha
            
        # Update and clip weights to prevent explosion
        self.weights[action] += learning_rate * error * state[0]
        np.clip(self.weights, -10.0, 10.0, out=self.weights)

    def decay_epsilon(self, decay_rate):
        """Decay epsilon exponentially by decay_rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-decay_rate))

    def save(self, name):
        np.save(name, self.weights)

    def load(self, name):
        self.weights = np.load(f"{name}.npy")


# Use LinearAgent instead of DQNAgent for faster learning
DQNAgent = LinearAgent  # For backwards compatibility

if __name__ == "__main__":
    print("This module is not meant to be run directly. Import it from a training script.")