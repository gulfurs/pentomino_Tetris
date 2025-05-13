import numpy as np
import random
from collections import deque
from constants import GRID_WIDTH, GRID_HEIGHT


class PentominoGameState:
    def __init__(self, game):
        self.game = game
        
    def get_state_matrix(self):
        # Create a matrix representing the current board state
        board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game.grid[y][x] is not None:
                    board[y, x] = 1
        
        for x, y in self.game.current_piece.get_coords():
            if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH:
                board[y, x] = 2
                
        return board
    
    def get_state_features(self):
        board = self.get_state_matrix()
        
        # distance from top
        column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                column_heights.append(0)
        
        # holes
        holes = 0
        for x in range(GRID_WIDTH):
            found_block = False
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    found_block = True
                elif found_block:
                    holes += 1
        
        aggregate_height = sum(column_heights)

        bumpiness = sum(abs(column_heights[i] - column_heights[i-1]) for i in range(1, GRID_WIDTH))
        
        complete_lines = 0
        near_complete_lines = 0  
        nearly_nearly_complete_lines = 0  
        for y in range(GRID_HEIGHT):
            filled_count = sum(1 for x in range(GRID_WIDTH) if board[y, x] > 0)
            if filled_count == GRID_WIDTH:
                complete_lines += 1
            elif filled_count >= GRID_WIDTH - 2:  
                near_complete_lines += 1
            elif filled_count >= GRID_WIDTH - 4:  
                nearly_nearly_complete_lines += 0.5  

        wells = 0
        for x in range(GRID_WIDTH):
            left_h = column_heights[x-1] if x > 0 else 0
            right_h = column_heights[x+1] if x < GRID_WIDTH-1 else 0
            current_h = column_heights[x]
            
            if current_h < left_h - 1 and current_h < right_h - 1:
                wells += min(left_h, right_h) - current_h
        
        piece_value = 0
        if self.game.current_piece.shape_name == 'I': 
            piece_value = 1.0
        elif self.game.current_piece.shape_name in ['L', 'J', 'T']: 
            piece_value = 0.7
        elif self.game.current_piece.shape_name == 'Z':  
            piece_value = 0.3
            
        row_transitions = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH - 1):
                if (board[y, x] > 0) != (board[y, x+1] > 0):
                    row_transitions += 1
                    
        col_transitions = 0
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT - 1):
                if (board[y, x] > 0) != (board[y+1, x] > 0):
                    col_transitions += 1
        
        max_height = max(column_heights) if column_heights else 0
        avg_height = aggregate_height / GRID_WIDTH if column_heights else 0
        
        current_piece_x = self.game.current_piece.x
        center_distance = abs(current_piece_x - GRID_WIDTH // 2) / GRID_WIDTH
        
        normalized_holes = holes / 20.0  
        normalized_bumpiness = min(bumpiness / 40.0, 1.0)  
        normalized_height = max_height / GRID_HEIGHT
        normalized_transitions = min((row_transitions + col_transitions) / 100.0, 1.0)
        
        features = np.array([
            normalized_holes * 2.0,                 # Holes
            normalized_bumpiness * 1.5,             # Bumpiness
            normalized_height,                      # Max height
            complete_lines * 3.0,                   # Complete lines
            near_complete_lines * 2.0,              # Near-complete lines
            wells * 0.5,                            # Wells
            avg_height / GRID_HEIGHT,               # Average height
            center_distance,                        # Distance from center
            piece_value,                            # Current piece value
            nearly_nearly_complete_lines             # Somewhat filled lines
        ])
        
        return features


class LinearAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.weights = np.random.uniform(-0.01, 0.01, (action_size, state_size))
        
        # Feature Holes
        self.weights[:, 0] = -0.9
        
        # Feature 1: Bumpiness
        self.weights[:, 1] = -0.4
        
        # Feature 2: Max height
        self.weights[:, 2] = -1.1
        
        # Feature 3: Complete lines
        self.weights[:, 3] = 3.0
        
        # Feature 4: Near-complete lines
        self.weights[:, 4] = 1.0
        
        # Feature 5: Wells 
        self.weights[:, 5] = 0.7
        
        # Feature 6: Average height
        self.weights[:, 6] = -0.2
        
        # Feature 7: Center distance
        self.weights[:, 7] = -0.01
        
        # Feature 8: Piece value 
        self.weights[:, 8] = 0.8
        
        # Feature 9: Nearly-nearly complete lines
        self.weights[:, 9] = 0.3
        
        self.weights[0, :] *= 1.2  # Left
        self.weights[1, :] *= 1.2  # Right
        self.weights[2, :] *= 1.1  # Rotate
        self.weights[3, :] *= 0.9  # Soft drop
        self.weights[4, :] *= 0.7  # Hard drop 

        # Learning parameters 
        self.alpha = 0.005  
        self.gamma = 0.95  
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        
        self.line_clear_actions = [0] * action_size
        self.total_actions = [0] * action_size
    
    def predict(self, state):
        return np.dot(self.weights, state[0])

    def act(self, state):
        if np.random.rand() < self.epsilon:
            total_actions = sum(self.total_actions)
            if total_actions > 100:  # Once we have some data

                success_rates = [(self.line_clear_actions[i] / max(self.total_actions[i], 1)) 
                               for i in range(self.action_size)]

                probs = [0.1 + 0.9 * rate for rate in success_rates]
                probs = [p/sum(probs) for p in probs]
                return np.random.choice(self.action_size, p=probs)
            else:
                probs = [0.4, 0.4, 0.1, 0.05, 0.05] 
                return np.random.choice(self.action_size, p=probs)
        
        # Otherwise, choose the best predicted action
        q_values = self.predict(state)
        return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
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
        
        self.total_actions[action] += 1
        if reward > 1000:  # Big reward indicates line clear
            self.line_clear_actions[action] += 1
        
        if reward > 1000:
            learning_rate = self.alpha * 10
        elif reward > 0:
            learning_rate = self.alpha * 2
        else:
            learning_rate = self.alpha
            
        self.weights[action] += learning_rate * error * state[0]
        
        np.clip(self.weights, -20.0, 20.0, out=self.weights)

    def decay_epsilon(self, decay_rate):
        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-decay_rate))

    def save(self, name):
        np.save(name, self.weights)

    def load(self, name):
        self.weights = np.load(f"{name}.npy")


DQNAgent = LinearAgent  #

if __name__ == "__main__":
    print("This is a module for a linear agent in a Tetris-like game.")