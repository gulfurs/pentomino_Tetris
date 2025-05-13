import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from rl_agent import PentominoGameState, LinearAgent
from game import PentominoGame
from constants import GRID_WIDTH, GRID_HEIGHT
import random

class GameWrapper:
    def __init__(self):
        self.game = PentominoGame(headless=True)
        self.game.render_enabled = False

    def reset(self):
        self.game.reset_game()
        state_obj = PentominoGameState(self.game)
        return state_obj.get_state_features().reshape(1, -1)

    def step(self, action):
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

        lines_cleared = self.game.lines_cleared - initial_lines
        reward = 0
        
        if lines_cleared > 0:
            reward += 2000 * (2 ** lines_cleared)  # 2000, 4000, 8000, 16000 for 1-4 lines
            
        holes = self.count_holes()
        reward -= holes * 2 
        
        # Height management
        heights = self.calculate_column_heights()
        max_height = max(heights) if heights else 0
        if max_height > GRID_HEIGHT // 2:
            # Progressive penalty as height increases
            penalty_factor = (max_height - GRID_HEIGHT // 2) ** 1.5
            reward -= penalty_factor * 5
            
        bumpiness = sum(abs(heights[i] - heights[i-1]) for i in range(1, len(heights)))
        reward -= bumpiness * 2  # Increased penalty for uneven surfaces
        
        reward -= 1
        
        if not self.game.game_over:
            reward += 0.15  #for survival
            
        done = self.game.game_over
        
        self.lines_cleared_episode = getattr(self, 'lines_cleared_episode', 0) + lines_cleared
        
        state_obj = PentominoGameState(self.game)
        next_state = state_obj.get_state_features().reshape(1, -1)
        
        return next_state, reward, done, {"lines": lines_cleared}

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

    def count_almost_complete_lines(self):
        almost = 0
        for y in range(GRID_HEIGHT):
            count = sum(1 for x in range(GRID_WIDTH) if self.game.grid[y][x] is not None)
            if GRID_WIDTH - 2 <= count < GRID_WIDTH:
                almost += 1
        return almost

def train_agent(episodes=1000, print_every=10):
    env = GameWrapper()
    state_size = 10
    action_size = 5
    agent = LinearAgent(state_size, action_size)

    EPSILON_START = 1.0
    EPSILON_END = 0.05
    decay_rate = -np.log(EPSILON_END / EPSILON_START) / (episodes * 5)  

    scores = []
    lines_cleared = []
    survival_steps = []  
    epsilons = []
    reward_history = deque(maxlen=50)
    lines_history = deque(maxlen=50)
    
    cumulative_rewards = []
    total_cumulative_reward = 0
    step_rewards = []  
    
    memory = deque(maxlen=10000)
    batch_size = 124
    
    os.makedirs('models', exist_ok=True)

    print(f"Starting training for {episodes} episodes")
    
    for ep in range(episodes):
        env.lines_cleared_episode = 0
        
        agent.epsilon = max(EPSILON_END, EPSILON_START * np.exp(-decay_rate * ep))

        state = env.reset()
        
        done = False
        total_reward = 0
        start_score = env.game.score
        episode_line_clears = 0
        training_steps = 0
        
        while not done:
            if np.random.rand() < agent.epsilon:
                action = np.random.choice(action_size)
            else:
                action = np.argmax(agent.predict(state))
            
            next_state, reward, done, info = env.step(action)
            
            if 'lines' in info and info['lines'] > 0:
                episode_line_clears += info['lines']
            
            memory.append((state, action, reward, next_state, done))
            
            agent.learn(state, action, reward, next_state, done)
            
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, ns, d in minibatch:
                    agent.learn(s, a, r, ns, d)
            
            state = next_state
            total_reward += reward
            total_cumulative_reward += reward
            step_rewards.append(reward)
            training_steps += 1

        episode_score = env.game.score - start_score
        episode_lines = env.lines_cleared_episode if hasattr(env, 'lines_cleared_episode') else 0
        
        #stats
        scores.append(episode_score)
        lines_cleared.append(episode_lines)
        survival_steps.append(training_steps)
        epsilons.append(agent.epsilon)
        reward_history.append(total_reward)
        lines_history.append(episode_lines)
        cumulative_rewards.append(total_cumulative_reward)
        
        if ep % print_every == 0 or episode_lines > 0:  
            avg_reward = np.mean(reward_history) if reward_history else 0
            avg_lines = np.mean(lines_history) if lines_history else 0
            print(f"Ep {ep}/{episodes} | Reward: {total_reward:.1f} | Score: {episode_score} | "
                  f"Lines: {episode_lines} | Steps: {training_steps} | Avg Lines: {avg_lines:.2f} | Epsilon: {agent.epsilon:.3f}")

    #final model
    agent.save('models/agent_final')
    print('Training complete. Final model saved.')
    

    # Plotting
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.plot(scores)
    plt.title('Score per Episode')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    
    plt.subplot(222)
    plt.plot(lines_cleared)
    plt.title('Lines Cleared per Episode')
    plt.ylabel('Lines')
    plt.xlabel('Episode')
    
    plt.subplot(223)
    plt.plot(epsilons)
    plt.title('Exploration Rate')
    plt.ylabel('Epsilon')
    plt.xlabel('Episode')
    
    plt.subplot(224)
    plt.plot(survival_steps)
    plt.title('Survival Time per Episode')
    plt.ylabel('Steps')
    plt.xlabel('Episode')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.plot(range(len(cumulative_rewards)), cumulative_rewards)
    plt.title('Cumulative Reward vs. Episode')
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Episode')
    
    if len(step_rewards) > 0:
        window_size = min(100, len(step_rewards) // 10)
        if window_size > 0:
            smoothed_rewards = np.convolve(step_rewards, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
            plt.subplot(122)
            plt.plot(range(len(smoothed_rewards)), smoothed_rewards)
            plt.title('Reward per Training Step (Smoothed)')
            plt.ylabel('Reward')
            plt.xlabel('Training Step')
    
    plt.tight_layout()
    plt.savefig('training_progress_improved.png')
    plt.show()
    
    plt.figure(figsize=(10, 5))
    window_size = min(50, episodes // 10) 
    if window_size > 0:
        moving_avg_steps = np.convolve(survival_steps, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
        plt.plot(range(len(moving_avg_steps)), moving_avg_steps)
        plt.title('Average Survival Time (Moving Average)')
        plt.ylabel('Steps (Moving Average)')
        plt.xlabel('Episode')
        plt.savefig('survival_time_trend.png')
        plt.show()
    
    return agent

if __name__ == '__main__':
    import sys
    episodes = 1000
    
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of episodes: {sys.argv[1]}, using default: 1000")
    
    trained_agent = train_agent(episodes=episodes, print_every=10)