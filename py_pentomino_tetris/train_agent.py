"""
Simplified training script for the Pentomino Tetris reinforcement learning agent
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
from rl_agent import PentominoGameState, LinearAgent
from game import PentominoGame
from constants import GRID_WIDTH, GRID_HEIGHT

class GameWrapper:
    """Wrapper for the game to allow the agent to interact with it"""
    def __init__(self):
        self.game = PentominoGame(headless=True)  # Always use headless mode for training
        self.game.render_enabled = False  # Disable rendering by default

    def reset(self):
        """Reset the game state and return the initial state"""
        self.game.reset_game()
        state_obj = PentominoGameState(self.game)
        return state_obj.get_state_features().reshape(1, -1)

    def step(self, action):
        """Perform an action and return the next state, reward, and done flag"""
        initial_score = self.game.score
        
        # Get the state before action (for comparison later)
        prev_state_obj = PentominoGameState(self.game)
        prev_board = prev_state_obj.get_state_matrix()
        prev_piece_coords = self.game.current_piece.get_coords()
        prev_piece_x = self.game.current_piece.x
        prev_piece_y = self.game.current_piece.y
        
        # Get current board columns with non-zero height
        board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game.grid[y][x] is not None:
                    board[y, x] = 1

        # Calculate pre-action column heights
        prev_column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    prev_column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                prev_column_heights.append(0)
                
        # Calculate unevenness of columns before action
        prev_std_heights = np.std(prev_column_heights) if prev_column_heights else 0
        
        # Track pieces across the board width horizontally
        visited_columns = set()
        if hasattr(self.game, 'action_history'):
            for pos in self.game.action_history:
                x = pos[0]
                if 0 <= x < GRID_WIDTH:
                    visited_columns.add(x)
        
        # Apply the action
        if action == 0:
            self.game.move_piece(-1)  # Move left
        elif action == 1:
            self.game.move_piece(1)   # Move right
        elif action == 2:
            self.game.rotate_piece()  # Rotate
        elif action == 3:
            self.game.move_piece_down()  # Soft drop
        elif action == 4:
            self.game.drop_piece()  # Hard drop
            
        # Track the current piece position for the action history
        if hasattr(self.game, 'action_history'):
            self.game.action_history.append((self.game.current_piece.x, self.game.current_piece.y))
            self.game.episode_actions += 1
            
        # Calculate rewards
        score = self.game.score - initial_score
        lines_cleared = self.game.lines_cleared
        
        # Calculate advanced metrics
        current_holes = self.calculate_holes(board)
        current_heights = self.calculate_column_heights(board)
        current_bumpiness = self.calculate_bumpiness(current_heights)
        current_max_height = max(current_heights) if current_heights else 0
        current_avg_height = sum(current_heights) / len(current_heights) if current_heights else 0
        current_height_variance = self.calculate_height_variance(current_heights)
        
        # Calculate center of gravity and its distance from ideal
        current_cog = self.calculate_center_of_gravity(board)
        
        # Enhanced vertical stacking detection and penalties
        vertical_stack_penalty = 0
        if current_max_height > 0:
            # Calculate vertical stack factor: how much taller is max height compared to average
            vertical_stack_factor = current_max_height / (current_avg_height + 0.1)  # Avoid div by zero
            
            # Progressive penalty based on vertical stack factor
            if vertical_stack_factor > 1.3:  # If max height is >30% above average, consider it vertical stacking
                # Exponential penalty for severe vertical stacking
                vertical_stack_penalty = (vertical_stack_factor - 1.3) ** 2 * 5.0
                
                # Additional penalty proportional to absolute height
                if current_max_height > GRID_HEIGHT / 2:  # If stack is more than half the grid height
                    tall_stack_factor = current_max_height / GRID_HEIGHT
                    vertical_stack_penalty += tall_stack_factor * 8.0
        
        # Calculate horizontal distribution of pieces
        filled_columns = sum(1 for height in current_heights if height > 0)
        horizontal_distribution = filled_columns / GRID_WIDTH if GRID_WIDTH > 0 else 0
        
        # Enhanced horizontal utilization reward
        horizontal_reward = horizontal_distribution * 2.0
        
        # Special reward for good horizontal distribution
        if horizontal_distribution > 0.7:  # Using more than 70% of columns
            horizontal_reward += 4.0
        elif horizontal_distribution > 0.5:  # Using more than 50% of columns
            horizontal_reward += 2.0
            
        # Strong penalties for holes, bumpiness and height
        hole_penalty = current_holes * 0.8
        bumpiness_penalty = current_bumpiness * 0.5
        height_penalty = current_max_height * 0.4
        
        # Calculate flat top bonus - we want a relatively flat surface
        if current_bumpiness < 3 and current_holes == 0:
            flat_top_bonus = 2.0
        elif current_bumpiness < 5:
            flat_top_bonus = 1.0
        else:
            flat_top_bonus = 0
        
        # Enhanced reward for lines cleared
        lines_reward = 0
        if lines_cleared == 1:
            lines_reward = 5.0
        elif lines_cleared == 2:
            lines_reward = 12.0
        elif lines_cleared == 3:
            lines_reward = 25.0
        elif lines_cleared >= 4:
            lines_reward = 50.0
        
        # Extreme penalty for game over
        game_over_penalty = 20.0 if self.game.game_over else 0
        
        # Extra rewards for better board structure to avoid high stacking
        structure_reward = 0
        if current_holes == 0:
            # Perfect board with no holes
            structure_reward += 2.0
            
        # Reward for keeping height low
        if current_max_height < GRID_HEIGHT / 4:  # Less than 25% of grid height
            structure_reward += 2.0
        elif current_max_height < GRID_HEIGHT / 2:  # Less than 50% of grid height
            structure_reward += 1.0
            
        # Height variance penalty - we want all columns to be similar height
        variance_penalty = current_height_variance * 0.3
        
        # Combined reward calculation with stronger emphasis on anti-vertical stacking
        reward = (
            score / 10.0 +
            lines_reward -
            hole_penalty -
            bumpiness_penalty - 
            height_penalty - 
            variance_penalty +
            flat_top_bonus +
            horizontal_reward +
            structure_reward -
            vertical_stack_penalty -  # Now much stronger
            game_over_penalty
        )
        
        done = self.game.game_over
        state_obj = PentominoGameState(self.game)
        next_state = state_obj.get_state_features().reshape(1, -1)
        
        # Include info about game state for statistics
        info = {
            "score": self.game.score,
            "level": self.game.level,
            "height": current_max_height,
            "holes": current_holes
        }
        
        return next_state, reward, done, info

    def calculate_holes(self, board):
        """Calculate the number of holes in the board"""
        holes = 0
        for x in range(GRID_WIDTH):
            found_block = False
            for y in range(GRID_HEIGHT):
                if board[y, x] == 1:
                    found_block = True
                elif found_block and board[y, x] == 0:
                    holes += 1
        return holes

    def calculate_column_heights(self, board):
        """Calculate the heights of each column"""
        column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if board[y, x] > 0:
                    column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                column_heights.append(0)
        return column_heights

    def calculate_bumpiness(self, column_heights):
        """Calculate the bumpiness of the board"""
        return sum(abs(column_heights[i] - column_heights[i-1]) for i in range(1, len(column_heights)))

    def calculate_height_variance(self, column_heights):
        """Calculate the variance of column heights"""
        return np.var(column_heights)

    def calculate_center_of_gravity(self, board):
        """Calculate the center of gravity of the board"""
        total_weight = 0
        total_x = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if board[y, x] > 0:
                    total_weight += 1
                    total_x += x
        return total_x / total_weight if total_weight > 0 else 0


def train_agent(episodes=500, print_every=10, save_every=50, show_plots=True):
    """Train the agent with simplified logic"""
    start_time = time.time()
    env = GameWrapper()
    state_size = 10  # Updated: Number of features in the state, now with enhanced features
    action_size = 5  # Number of possible actions
    agent = LinearAgent(state_size, action_size)
    
    # For tracking progress
    scores = []
    avg_scores = []
    epsilons = []
    episodes_times = []
    max_heights = []
    holes_counts = []
    
    print("Starting training...")
    
    for episode in range(episodes):
        episode_start = time.time()
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Track episode number in agent for epsilon decay control
        agent.episodes_seen = episode + 1
        
        # Fast game loop (no rendering)
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            # Avoid infinite loops
            if steps > 10000:
                print("Episode taking too long, terminating...")
                break
        
        # Track performance for adaptive learning
        agent.track_performance(info["score"])
        
        # Record stats
        episode_time = time.time() - episode_start
        episodes_times.append(episode_time)
        scores.append(info["score"])
        avg_scores.append(np.mean(scores[-min(len(scores), 100):]))
        epsilons.append(agent.epsilon)
        max_heights.append(info["height"])
        holes_counts.append(info["holes"])
        
        # Print progress
        if episode % print_every == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
            print(f"Episode {episode+1}/{episodes} | Score: {info['score']} | " +
                  f"Avg Score: {avg_scores[-1]:.1f} | Level: {info['level']} | " +
                  f"Epsilon: {agent.epsilon:.3f} | Reward: {total_reward:.1f} | " +
                  f"Height: {info['height']} | Holes: {info['holes']} | " +
                  f"Time: {episode_time:.2f}s | Speed: {eps_per_sec:.1f} eps/s")
        
        # Save periodically
        if episode % save_every == 0 and episode > 0:
            agent.save(f"models/agent_episode_{episode}")
            # Also print the weights so we can see what it's learning
            # agent.print_weights()
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    agent.save("models/agent_final")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTraining complete in {total_time:.1f} seconds!")
    print(f"Final average score: {avg_scores[-1]:.1f}")
    print(f"Best score: {max(scores)}")
    
    # Plot training progress
    if show_plots:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(scores)
        plt.title('Score per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(2, 3, 2)
        plt.plot(avg_scores)
        plt.title('Average Score (last 100 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        plt.subplot(2, 3, 3)
        plt.plot(epsilons)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.subplot(2, 3, 4)
        plt.plot(episodes_times)
        plt.title('Episode Duration')
        plt.xlabel('Episode')
        plt.ylabel('Duration (seconds)')
        
        plt.subplot(2, 3, 5)
        plt.plot(max_heights)
        plt.title('Max Stack Height')
        plt.xlabel('Episode')
        plt.ylabel('Height')
        
        plt.subplot(2, 3, 6)
        plt.plot(holes_counts)
        plt.title('Holes Count')
        plt.xlabel('Episode')
        plt.ylabel('Holes')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
    
    return agent


if __name__ == "__main__":
    try:
        import matplotlib
        # Check if running in headless environment
        if not os.environ.get('DISPLAY') and os.name != 'nt':
            matplotlib.use('Agg')  # Use non-interactive backend if no display
            show_plots = False
        else:
            show_plots = True
    except ImportError:
        show_plots = False
        print("Matplotlib not installed. Skipping plots.")
    
    # Get training parameters
    try:
        episodes = int(input("Enter number of episodes to train (default: 100): ") or "100")
    except ValueError:
        episodes = 100
        
    # Train the agent
    agent = train_agent(episodes=episodes, show_plots=show_plots)
    
    # Make the playback easier to run
    print("\nTo watch the trained agent play, run: python playback.py")