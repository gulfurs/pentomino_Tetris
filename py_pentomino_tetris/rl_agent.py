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
    Simple linear function approximator for Q-learning.
    Much faster than Deep Q-Network for simple problems.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize weights with small random values instead of zeros
        # This helps break symmetry and speeds up learning
        self.weights = np.random.uniform(-0.1, 0.1, (action_size, state_size))
        
        # Initialize with stronger domain knowledge to discourage vertical stacking:
        for a in range(action_size):
            # Strongly penalize holes
            self.weights[a][0] = -0.2
            # Strongly penalize bumpiness
            self.weights[a][1] = -0.15
            # Severely penalize height
            self.weights[a][2] = -0.25
            # Reward line clears
            self.weights[a][3] = 0.2
            # Reward near-complete lines
            self.weights[a][4] = 0.15
            # Reward line opportunities
            self.weights[a][5] = 0.15
            # Penalize average height
            self.weights[a][6] = -0.2
            
        # Much stronger horizontal movement bias
        self.weights[0][7] = -0.2  # Move left when far from center on right
        self.weights[1][7] = 0.2  # Move right when far from center on left
        
        # Better reward for rotation which helps fit pieces
        self.weights[2][1] = -0.1  # Rotation should reduce bumpiness
        self.weights[2][4] = 0.1   # Rotation that creates opportunities for near-complete lines
        
        # Smarter hard drop logic - only drop when it makes sense
        self.weights[4][5] = 0.15  # Hard drop on good line opportunities
        self.weights[4][2] = -0.2  # Really discourage hard drops that increase height
        self.weights[4][0] = -0.2  # Really discourage hard drops that create holes
        
        # Learning parameters - adjusted for better exploration and learning
        self.alpha = 0.03  # Higher learning rate
        self.gamma = 0.99  # Higher discount factor for long-term rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # More gradual decay
        self.min_explore_episodes = 80  # Extended exploration period
        self.episodes_seen = 0  # Track episodes for exploration control
        
        # Experience buffer
        self.memory = deque(maxlen=7500)  # Larger memory for better learning
        
        # Performance tracking
        self.best_score = 0
        self.best_weights = None  # Store weights that achieved best score
        self.last_improve = 0  # Episode of last improvement
        self.no_improve_count = 0  # Count episodes with no improvement
        self.training_iterations = 0
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        # Store the experience
        self.memory.append((state, action, reward, next_state, done))
        
        # Learn more frequently from experiences
        if done or len(self.memory) % 8 == 0:  # Learn every 8 steps or on episode end
            self._learn_from_memory(min(len(self.memory), 96))  # Bigger batch size
    
    def _learn_from_memory(self, batch_size):
        """Learn from past experiences"""
        if len(self.memory) < batch_size:
            return
            
        # Sample with preference for recent experiences (helps with non-stationarity)
        if len(self.memory) > batch_size*2:
            # 75% recent experiences, 25% random from all experiences
            recent_size = int(batch_size * 0.75)
            random_size = batch_size - recent_size
            
            recent_batch = list(self.memory)[-recent_size:]
            random_batch = random.sample(list(self.memory), random_size)
            batch = recent_batch + random_batch
        else:
            batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Extract feature vectors
            state_features = state[0]  # Remove batch dimension
            next_state_features = next_state[0]  # Remove batch dimension
            
            # Calculate target: r + γ·max Q(s',a')
            if done:
                target = reward
            else:
                # Calculate Q values for next state and find the maximum
                next_q_values = np.dot(self.weights, next_state_features)
                target = reward + self.gamma * np.max(next_q_values)
            
            # Calculate current Q value
            current_q = np.dot(self.weights[action], state_features)
            
            # Update weights for the selected action
            # w = w + α·(target - current_q)·features
            # Use a varying learning rate - larger for bigger errors
            error = target - current_q
            adaptive_alpha = self.alpha * (1.0 + 0.6 * abs(error))  # Adapt based on error size
            self.weights[action] += adaptive_alpha * error * state_features
        
        # Only decay epsilon after minimum exploration episodes
        if self.episodes_seen > self.min_explore_episodes and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_iterations += 1
    
    def act(self, state):
        """Choose an action based on the current state"""
        # Apply more gradual epsilon decay each time act() is called
        if self.episodes_seen > self.min_explore_episodes and self.epsilon > self.epsilon_min:
            # Apply tiny decay on each action for smoother transition
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.9998)
        
        # Special case for early episodes - force even more horizontal movement
        if self.episodes_seen < 90:  # Extended early learning phase
            # 80% chance for horizontal movement, 15% for rotation, 5% for other actions
            dice = random.random()
            if dice < 0.40:
                return 0  # Move left
            elif dice < 0.80:
                return 1  # Move right
            elif dice < 0.95:
                return 2  # Rotate
            elif dice < 0.98:
                return 3  # Soft drop
            else:
                return 4  # Hard drop
        
        # Normal exploration vs exploitation with stronger bias toward horizontal movement
        if np.random.rand() <= self.epsilon:
            # Even during normal exploration, heavily bias toward horizontal movement
            # 50% left, 50% right, 7% rotate, 2% soft drop, 1% hard drop
            dice = random.random()
            if dice < 0.45:
                return 0  # Move left
            elif dice < 0.90:
                return 1  # Move right
            elif dice < 0.97:
                return 2  # Rotate
            elif dice < 0.99:
                return 3  # Soft drop
            else:
                return 4  # Hard drop
        
        # Exploitation: select the action with highest Q-value
        state_features = state[0]  # Remove batch dimension
        q_values = np.dot(self.weights, state_features)
        
        # Add significant bias toward horizontal movement even during exploitation
        # This helps counteract the tendency to build vertical stacks
        q_values[0] += 0.08  # Stronger boost for left movement
        q_values[1] += 0.08  # Stronger boost for right movement
        
        # Penalize hard drops to discourage premature piece placement
        q_values[4] -= 0.04  # Stronger penalty for hard drop
        
        # Dynamically adjust action selection based on board state
        max_height = state_features[2] * GRID_HEIGHT  # Un-normalize the max height
        bumpiness = state_features[1] * 2  # Un-scale the bumpiness
        holes = state_features[0]
        
        # If the board is getting high, strongly discourage hard drops
        if max_height > GRID_HEIGHT/3:  # If the stack is getting high (>33% of grid height)
            height_factor = max_height / GRID_HEIGHT
            q_values[4] -= height_factor * 0.2  # Proportional penalty to hard drop
            
            # When board is high, favor horizontal movement even more
            q_values[0] += height_factor * 0.1
            q_values[1] += height_factor * 0.1
        
        # If there are many holes, favor rotation and horizontal movement
        if holes > 2:
            q_values[2] += holes * 0.03  # Encourage rotation to fit better
            q_values[4] -= holes * 0.03  # Discourage hard drop when holes exist
            
        # If bumpiness is high, strongly encourage horizontal movement and rotation
        if bumpiness > 3:  # If the board is very bumpy
            q_values[0] += bumpiness * 0.02  # Extra boost for horizontal movement
            q_values[1] += bumpiness * 0.02
            q_values[2] += bumpiness * 0.02  # Encourage rotation to flatten board
        
        # Check if we're nearing the middle of the board - try to avoid it
        center_distance = state_features[7]
        if center_distance < 0.3:  # Close to center
            # Encourage moving away from center
            q_values[0] += (0.3 - center_distance) * 0.15  # Move left
            q_values[1] += (0.3 - center_distance) * 0.15  # Move right
        
        # Add some small noise to break ties randomly
        q_values = q_values + np.random.uniform(-0.01, 0.01, self.action_size)
        
        # Special case: if we've been stacking vertically, force horizontal movement
        # Check for high columns and low average - indicates vertical stacking
        if max_height > 5 and max_height > state_features[6] * GRID_HEIGHT * 1.5:
            # We have a tall column - force horizontal movement
            horiz_values = q_values.copy()
            horiz_values[2:] = -999  # Strongly discourage non-horizontal moves
            return np.argmax(horiz_values)
        
        return np.argmax(q_values)
        
    def track_performance(self, score):
        """Track agent performance for adaptive learning"""
        if score > self.best_score:
            self.best_score = score
            self.best_weights = self.weights.copy()
            self.last_improve = self.episodes_seen
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            
            # If no improvement for a long time, try something different
            if self.no_improve_count > 40:  # Reduced from 50
                print(f"No improvement for {self.no_improve_count} episodes. Adjusting learning strategy.")
                if self.best_weights is not None:
                    # Blend current weights with best weights
                    self.weights = 0.8 * self.best_weights + 0.2 * self.weights  # More weight to best weights
                # Increase exploration temporarily
                self.epsilon = min(0.6, self.epsilon * 2)  # More exploration
                self.no_improve_count = 0
                
                # Reinforce anti-vertical stacking biases
                for a in range(self.action_size):
                    # Strengthen penalties for height and holes
                    self.weights[a][0] *= 1.1  # Increase hole penalty
                    self.weights[a][2] *= 1.1  # Increase height penalty
    
    def print_weights(self):
        """Print the learned weights for debugging"""
        print("\nLearned Feature Weights:")
        feature_names = ["Holes", "Bumpiness", "Max Height", "Complete Lines", 
                        "Near-complete Lines", "Line Opportunity", "Avg Height",
                        "Center Distance", "I-Piece", "Versatile Piece"]
        action_names = ["Move Left", "Move Right", "Rotate", "Soft Drop", "Hard Drop"]
        
        print("Feature weights by action:")
        for a in range(self.action_size):
            print(f"{action_names[a]}: ", end="")
            for f in range(self.state_size):
                print(f"{feature_names[f]}={self.weights[a][f]:.3f}  ", end="")
            print()
    
    def replay(self):
        """Empty implementation for compatibility with the main training loop"""
        pass  # Learning happens in remember()
    
    def save(self, name):
        """Save the learned weights"""
        # Save current weights
        np.save(name, self.weights)
        # If we have best weights, save those too
        if self.best_weights is not None:
            np.save(f"{name}_best", self.best_weights)
        print(f"Model saved to {name}.npy with weights shape {self.weights.shape}")
        
    def load(self, name):
        """Load the learned weights"""
        try:
            # Try to load best weights if available
            self.weights = np.load(f"{name}_best.npy")
            print(f"Model loaded from {name}_best.npy with weights shape {self.weights.shape}")
        except FileNotFoundError:
            self.weights = np.load(f"{name}.npy")
            print(f"Model loaded from {name}.npy with weights shape {self.weights.shape}")
        self.epsilon = self.epsilon_min  # Set to minimum exploration


# Use LinearAgent instead of DQNAgent for faster learning
DQNAgent = LinearAgent  # For backwards compatibility

if __name__ == "__main__":
    print("This module is not meant to be run directly. Import it from a training script.")