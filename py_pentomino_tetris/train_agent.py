import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from rl_agent import PentominoGameState, LinearAgent
from game import PentominoGame
from constants import GRID_WIDTH, GRID_HEIGHT, RED  # import RED for prefill blocks
import random  # for replay memory sampling

REWARD_BUMPINESS_PENALTY = 1  # Define penalty for bumpiness
HORIZ_EXPLORE_EPISODES = 100  # Extended period for horizontal exploration focus
PREFILL_EPISODES = 50  # Initial episodes to prefill near-complete lines for training

class GameWrapper:
    def __init__(self):
        self.game = PentominoGame(headless=True)
        self.game.render_enabled = False

    def reset(self, prefill=False):
        self.game.reset_game()
        # Setup a one-hole bottom row to guarantee a line clear
        self.randomize_board()
        self.force_initial_clear()
        if prefill:
            self.prefill_lines()
        state_obj = PentominoGameState(self.game)
        # Initialize horizontal fill for comparison
        self.prev_horizontal_fill = self.calculate_horizontal_fill()
        return state_obj.get_state_features().reshape(1, -1)

    def randomize_board(self):
        """
        Clear grid and create a full bottom row with exactly one hole for forced line clear
        """
        # Empty the grid
        self.game.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        # Select hole position in bottom row
        hole_x = random.randint(0, GRID_WIDTH - 1)
        self.special_hole = hole_x
        # Fill bottom row except at hole_x
        y = GRID_HEIGHT - 1
        for x in range(GRID_WIDTH):
            if x != hole_x:
                self.game.grid[y][x] = RED

    def force_initial_clear(self):
        """
        Drop a vertical I-piece into the bottom hole to clear the bottom row.
        """
        from pentomino import Pentomino
        hole = getattr(self, 'special_hole', None)
        if hole is None:
            return
        # Prepare a vertical I-piece at the hole x-position
        i_piece = Pentomino('I')
        i_piece.rotation = 0  # vertical orientation
        i_piece.x = hole
        i_piece.y = -4  # start above grid enough for 5-length piece
        self.game.current_piece = i_piece
        # Drop to lock and clear
        self.game.drop_piece()
        del self.special_hole

    def step(self, action):
        # Clone grid to compute new locked blocks for reward
        old_grid = [row.copy() for row in self.game.grid]
        initial_score = self.game.score
        initial_lines = self.game.lines_cleared

        # Take action
        if action == 0:
            self.game.move_piece(-1)
        elif action == 1:
            self.game.move_piece(1)
        elif action == 2:
            self.game.rotate_piece()
        elif action == 3:
            self.game.move_piece_down()
        elif action == 4:
            self.game.drop_piece()

        # Calculate reward - HEAVILY prioritize line clearing
        lines_cleared = self.game.lines_cleared - initial_lines
        reward = 0
        
        # Dramatically increase line clearing reward - this is our primary objective
        if lines_cleared > 0:
            # Exponential reward for lines cleared to prioritize this behavior
            reward += 2000 * (2 ** lines_cleared)  # 2000, 4000, 8000, 16000 for 1-4 lines
            
        # Hole penalty (severe) - holes are very detrimental
        holes = self.count_holes()
        reward -= holes * 20  # Increased penalty for holes
        
        # Height management - keep the stack low
        heights = self.calculate_column_heights()
        max_height = max(heights) if heights else 0
        if max_height > GRID_HEIGHT // 2:
            # Progressive penalty as height increases
            penalty_factor = (max_height - GRID_HEIGHT // 2) ** 1.5
            reward -= penalty_factor * 5
            
        # Evenness of the surface (reduces bumpiness)
        bumpiness = sum(abs(heights[i] - heights[i-1]) for i in range(1, len(heights)))
        reward -= bumpiness * 2  # Increased penalty for uneven surfaces
        
        # Bonus for creating "well-structured" boards (good for Tetris)
        # Reward horizontally filled rows (near-complete lines)
        near_complete = self.count_almost_complete_lines()
        reward += near_complete * 50  # Big bonus for setting up potential line clears
        
        # Small step penalty to encourage efficiency
        reward -= 1
        
        # Check for new piece placement (immediate feedback)
        new_blocks = sum(1 for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH)
                         if old_grid[y][x] is None and self.game.grid[y][x] is not None)
        if new_blocks > 0:
            # Small bonus for successfully placing pieces
            reward += 5

        # Game state update
        done = self.game.game_over
        
        # Track lines cleared directly in the wrapper for better monitoring
        self.lines_cleared_episode = getattr(self, 'lines_cleared_episode', 0) + lines_cleared
        
        # Get the next state
        state_obj = PentominoGameState(self.game)
        next_state = state_obj.get_state_features().reshape(1, -1)
        
        return next_state, reward, done, {"lines": lines_cleared}

    def prefill_lines(self, num_lines=3, missing_per_line=2):
        """
        Prefill bottom rows with near-complete lines to encourage learning line clears
        """
        import random
        for i in range(num_lines):
            y = GRID_HEIGHT - 1 - i
            missing = set(random.sample(range(GRID_WIDTH), missing_per_line))
            for x in range(GRID_WIDTH):
                if x not in missing:
                    # fill with a dummy block color
                    self.game.grid[y][x] = RED

    def calculate_column_heights(self):
        column_heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x] is not None:
                    column_heights.append(GRID_HEIGHT - y)
                    break
            else:
                column_heights.append(0)
        return column_heights

    def count_holes(self):
        holes = 0
        for x in range(GRID_WIDTH):
            block_found = False
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x] is not None:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def calculate_horizontal_fill(self):
        filled_counts = []
        for y in range(GRID_HEIGHT):
            count = sum(1 for x in range(GRID_WIDTH) if self.game.grid[y][x] is not None)
            filled_counts.append(count / GRID_WIDTH)
        return sum(filled_counts) / len(filled_counts)

    def count_almost_complete_lines(self):
        almost = 0
        for y in range(GRID_HEIGHT):
            count = sum(1 for x in range(GRID_WIDTH) if self.game.grid[y][x] is not None)
            if GRID_WIDTH - 2 <= count < GRID_WIDTH:
                almost += 1
        return almost

def train_agent(episodes=1000, print_every=10, save_every=100):
    env = GameWrapper()
    state_size = 10
    action_size = 5
    agent = LinearAgent(state_size, action_size)

    # Epsilon decay schedule parameters - start exploring heavily, then taper off
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    decay_rate = -np.log(EPSILON_END / EPSILON_START) / episodes

    scores = []
    lines_cleared = []
    epsilons = []
    reward_history = deque(maxlen=50)
    lines_history = deque(maxlen=50)
    
    # Experience replay buffer with prioritization for line clears
    memory = deque(maxlen=20000)
    priority_memory = deque(maxlen=5000)  # Special memory for successful line clears
    batch_size = 64
    
    # Create directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Best score tracking for model saving
    best_avg_lines = -float('inf')
    no_improvement_count = 0

    print(f"Starting training for {episodes} episodes...")
    print("Focus: Line clearing in Pentomino Tetris")
    
    for ep in range(episodes):
        # Reset the environment and tracking variables for this episode
        env.lines_cleared_episode = 0
        
        # Update epsilon using standard exponential decay
        agent.epsilon = max(EPSILON_END, EPSILON_START * np.exp(-decay_rate * ep))

        # Use prefill for the early episodes to enable line clears
        prefill = ep < PREFILL_EPISODES
        state = env.reset(prefill=prefill)
        
        # Initialize episode variables
        done = False
        total_reward = 0
        start_score = env.game.score
        start_lines = env.game.lines_cleared
        episode_line_clears = 0  # Track line clears directly in this episode
        
        # Force an initial drop to possibly clear the prepared line
        initial_state = state
        next_state, init_reward, done, info = env.step(4)  # hard drop action
        if 'lines' in info and info['lines'] > 0:
            episode_line_clears += info['lines']
        agent.learn(initial_state, 4, init_reward, next_state, done)
        state = next_state
        total_reward += init_reward

        while not done:
            # Action selection strategy based on training phase
            if ep < HORIZ_EXPLORE_EPISODES:
                # Early episodes: focus on horizontal exploration
                probs = [0.4, 0.4, 0.1, 0.05, 0.05]  # Left, Right, Rotate, Down, Drop
                action = np.random.choice(action_size, p=probs)
            elif np.random.rand() < agent.epsilon:
                # Exploration phase with smart bias
                if ep < episodes // 2:  # First half of training
                    # Strong bias against hard drops early on
                    probs = [0.35, 0.35, 0.2, 0.08, 0.02]  # Left, Right, Rotate, Down, Drop (rarely)
                else:
                    # More balanced once agent has learned basics
                    probs = [0.25, 0.25, 0.2, 0.15, 0.15]  # More balanced
                action = np.random.choice(action_size, p=probs)
            else:
                # Exploitation: choose best action based on learned values
                action = np.argmax(agent.predict(state))
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Track line clears directly from info
            if 'lines' in info and info['lines'] > 0:
                episode_line_clears += info['lines']
                
                # Add to priority memory if lines were cleared (to revisit successful actions)
                priority_memory.append((state, action, reward, next_state, done))
            
            # Store transition in regular memory
            memory.append((state, action, reward, next_state, done))
            
            # Online learning - immediate update
            agent.learn(state, action, reward, next_state, done)
            
            # Experience replay - balanced between regular and priority memories
            if len(memory) >= batch_size:
                # Prioritize line-clearing experiences in replay
                if len(priority_memory) > batch_size // 4 and np.random.random() < 0.5:
                    # Sample from priority memory (successful line clears)
                    minibatch = random.sample(priority_memory, min(batch_size // 4, len(priority_memory)))
                    # Complete batch with regular experiences
                    if len(memory) > batch_size - len(minibatch):
                        minibatch.extend(random.sample(memory, batch_size - len(minibatch)))
                else:
                    # Sample from regular memory
                    minibatch = random.sample(memory, batch_size)
                    
                # Learn from batch
                for s, a, r, ns, d in minibatch:
                    agent.learn(s, a, r, ns, d)
            
            state = next_state
            total_reward += reward

        # End of episode tracking
        episode_score = env.game.score - start_score
        episode_lines = env.lines_cleared_episode if hasattr(env, 'lines_cleared_episode') else 0
        
        # Double-check we're tracking lines correctly - prevent negative line counts
        if episode_lines < 0:  # Safeguard against potential bugs in line tracking
            episode_lines = episode_line_clears  # Use our direct counter instead
        
        # Store episode stats
        scores.append(episode_score)
        lines_cleared.append(episode_lines)
        epsilons.append(agent.epsilon)
        reward_history.append(total_reward)
        lines_history.append(episode_lines)
        
        # Print progress
        if ep % print_every == 0 or episode_lines > 0:  # Always print when lines are cleared
            avg_reward = np.mean(reward_history) if reward_history else 0
            avg_lines = np.mean(lines_history) if lines_history else 0
            print(f"Ep {ep}/{episodes} | Reward: {total_reward:.1f} | Score: {episode_score} | "
                  f"Lines: {episode_lines} | Avg Lines: {avg_lines:.2f} | Epsilon: {agent.epsilon:.3f}")
        
        # Save intermediate models
        if ep % save_every == 0 and ep > 0:
            agent.save(f"models/agent_episode_{ep}")
            print(f"Checkpoint saved at episode {ep}")
            
        # Save best model based on line clearing performance
        if len(lines_history) >= 20:  # Wait until we have enough history
            current_avg_lines = np.mean(lines_history)
            if current_avg_lines > best_avg_lines:
                best_avg_lines = current_avg_lines
                agent.save("models/agent_best")
                print(f"New best model saved! Avg Lines: {best_avg_lines:.2f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # If no improvement for a long time, adjust learning rate
            if no_improvement_count > 100:
                agent.alpha *= 0.9  # Reduce learning rate
                print(f"No improvement for {no_improvement_count} episodes. Adjusting learning rate to {agent.alpha:.5f}")
                no_improvement_count = 0

    # Save final model
    agent.save('models/agent_final')
    print('Training complete. Final model saved.')
    
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    
    plt.subplot(132)
    plt.plot(lines_cleared)
    plt.title('Lines Cleared per Episode')
    plt.ylabel('Lines')
    plt.xlabel('Episode')
    
    plt.subplot(133)
    plt.plot(epsilons)
    plt.title('Exploration Rate')
    plt.ylabel('Epsilon')
    plt.xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig('training_progress_improved.png')
    plt.show()
    
    return agent

if __name__ == '__main__':
    # Allow command-line specification of episodes
    import sys
    episodes = 1000
    
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of episodes: {sys.argv[1]}, using default: 1000")
    
    trained_agent = train_agent(episodes=episodes, print_every=10, save_every=100)