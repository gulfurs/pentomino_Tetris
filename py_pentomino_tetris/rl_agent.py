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
        Extract relevant features from the game state with strong emphasis on line clearing potential
        
        Returns:
            np.array: A feature vector optimized for line clearing training
        """
        board = self.get_state_matrix()
        
        # Calculate column heights (distance from top)
        column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                column_heights.append(0)
        
        # Number of holes (empty cells with filled cells above them) - severe penalty
        holes = 0
        for x in range(GRID_WIDTH):
            found_block = False
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    found_block = True
                elif found_block:
                    holes += 1
        
        # Calculate aggregate height (sum of all column heights)
        aggregate_height = sum(column_heights)
        
        # Calculate bumpiness (sum of height differences between adjacent columns)
        bumpiness = sum(abs(column_heights[i] - column_heights[i-1]) for i in range(1, GRID_WIDTH))
        
        # Count lines ready to clear
        complete_lines = 0
        near_complete_lines = 0  # Lines with 1-2 cells missing
        nearly_nearly_complete_lines = 0  # Lines with 3-4 cells missing
        for y in range(GRID_HEIGHT):
            filled_count = sum(1 for x in range(GRID_WIDTH) if board[y, x] > 0)
            if filled_count == GRID_WIDTH:
                complete_lines += 1
            elif filled_count >= GRID_WIDTH - 2:  # Only 1-2 cells missing
                near_complete_lines += 1
            elif filled_count >= GRID_WIDTH - 4:  # 3-4 cells missing, still worth tracking
                nearly_nearly_complete_lines += 0.5  # Fractional value to reduce importance
        
        # Count wells (good for vertical pieces)
        wells = 0
        for x in range(GRID_WIDTH):
            left_h = column_heights[x-1] if x > 0 else 0
            right_h = column_heights[x+1] if x < GRID_WIDTH-1 else 0
            current_h = column_heights[x]
            
            if current_h < left_h - 1 and current_h < right_h - 1:
                wells += min(left_h, right_h) - current_h
        
        # Current piece type - special value for line-clearing pieces
        piece_value = 0
        if self.game.current_piece.shape_name == 'I':  # Best for clearing
            piece_value = 1.0
        elif self.game.current_piece.shape_name in ['L', 'J', 'T']:  # Good for clearing
            piece_value = 0.7
        elif self.game.current_piece.shape_name == 'Z':  # Worse for stacking
            piece_value = 0.3
            
        # Row transitions - changes between filled/empty cells horizontally
        # More transitions means less orderly board = bad
        row_transitions = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH - 1):
                if (board[y, x] > 0) != (board[y, x+1] > 0):
                    row_transitions += 1
                    
        # Column transitions - changes between filled/empty cells vertically
        col_transitions = 0
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT - 1):
                if (board[y, x] > 0) != (board[y+1, x] > 0):
                    col_transitions += 1
        
        # Max height and average height
        max_height = max(column_heights) if column_heights else 0
        avg_height = aggregate_height / GRID_WIDTH if column_heights else 0
        
        # Current piece position - how centered is it (important for flexibility)
        current_piece_x = self.game.current_piece.x
        center_distance = abs(current_piece_x - GRID_WIDTH // 2) / GRID_WIDTH
        
        # Normalize values to similar scales to help learning
        normalized_holes = holes / 20.0  # Assuming max holes around 20
        normalized_bumpiness = min(bumpiness / 40.0, 1.0)  # Cap at 1.0
        normalized_height = max_height / GRID_HEIGHT
        normalized_transitions = min((row_transitions + col_transitions) / 100.0, 1.0)
        
        # Aggregate features - emphasize line clearing potential
        features = np.array([
            normalized_holes * 2.0,                 # Holes (very bad for line clearing)
            normalized_bumpiness * 1.5,             # Bumpiness (bad for stacking)
            normalized_height,                      # Max height (high board is risky)
            complete_lines * 3.0,                   # Complete lines (major goal)
            near_complete_lines * 2.0,              # Near-complete lines (valuable)
            wells * 0.5,                            # Wells (good for I-pieces)
            avg_height / GRID_HEIGHT,               # Average height (keep low)
            center_distance,                        # Distance from center (bad)
            piece_value,                            # Current piece value (I-piece is best)
            nearly_nearly_complete_lines             # Somewhat filled lines (minor value)
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
        
        # Initialize weights with proper bias for Tetris-style games
        self.weights = np.random.uniform(-0.01, 0.01, (action_size, state_size))
        
        # Strongly bias the weights toward optimal Tetris strategy
        # Feature 0: Holes (very bad)
        self.weights[:, 0] = -1.0
        
        # Feature 1: Bumpiness (bad)
        self.weights[:, 1] = -0.5
        
        # Feature 2: Max height (bad when too high)
        self.weights[:, 2] = -0.3
        
        # Feature 3: Complete lines (very good) - highest priority
        self.weights[:, 3] = 3.0
        
        # Feature 4: Near-complete lines (good)
        self.weights[:, 4] = 1.0
        
        # Feature 5: Wells (good for I-pieces)
        self.weights[:, 5] = 0.7
        
        # Feature 6: Average height (should be moderate)
        self.weights[:, 6] = -0.2
        
        # Feature 7: Center distance (keep pieces centered)
        self.weights[:, 7] = -0.5
        
        # Feature 8: Piece value (I-piece is best)
        self.weights[:, 8] = 0.8
        
        # Feature 9: Nearly-nearly complete lines (minor benefit)
        self.weights[:, 9] = 0.3
        
        # Bias actions differently to encourage exploration
        # Left/right movement should be favored over drops early on
        self.weights[0, :] *= 1.2  # Left
        self.weights[1, :] *= 1.2  # Right
        self.weights[2, :] *= 1.1  # Rotate
        self.weights[3, :] *= 0.9  # Soft drop
        self.weights[4, :] *= 0.7  # Hard drop (most punitive early on)

        # Learning parameters - increased learning rate for better adaptation
        self.alpha = 0.02  # Increased learning rate
        self.gamma = 0.95  # Slightly decreased discount factor for more immediate rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        
        # Keep track of successful actions for analytics
        self.line_clear_actions = [0] * action_size
        self.total_actions = [0] * action_size
    
    def predict(self, state):
        """Return Q-values for all actions"""
        return np.dot(self.weights, state[0])

    def act(self, state):
        """Epsilon-greedy action selection with bias toward actions that worked well"""
        if np.random.rand() < self.epsilon:
            # When exploring, use knowledge of good actions
            total_actions = sum(self.total_actions)
            if total_actions > 100:  # Once we have some data
                # Calculate success rates for each action
                success_rates = [(self.line_clear_actions[i] / max(self.total_actions[i], 1)) 
                               for i in range(self.action_size)]
                # Convert to probabilities (with some randomness)
                probs = [0.1 + 0.9 * rate for rate in success_rates]
                probs = [p/sum(probs) for p in probs]
                return np.random.choice(self.action_size, p=probs)
            else:
                # Until we have data, prefer horizontal movement
                probs = [0.4, 0.4, 0.1, 0.05, 0.05]  # Left, Right, Rotate, Soft Drop, Hard Drop
                return np.random.choice(self.action_size, p=probs)
        
        # Otherwise, choose the best predicted action
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
        
        # Track successful actions (those that lead to line clears)
        self.total_actions[action] += 1
        if reward > 1000:  # Big reward indicates line clear
            self.line_clear_actions[action] += 1
        
        # Dynamic learning rate - higher for significant positive rewards (line clears)
        if reward > 1000:
            # Line clear - boost learning rate significantly
            learning_rate = self.alpha * 10
        elif reward > 0:
            # Other positive rewards - moderate boost
            learning_rate = self.alpha * 2
        else:
            # Negative rewards - use base learning rate
            learning_rate = self.alpha
            
        # Update weights using the TD-error and learning rate
        self.weights[action] += learning_rate * error * state[0]
        
        # Ensure weights don't explode
        np.clip(self.weights, -20.0, 20.0, out=self.weights)

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