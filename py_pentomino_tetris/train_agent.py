import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque
from rl_agent import PentominoGameState, LinearAgent
from game import PentominoGame
from constants import GRID_WIDTH, GRID_HEIGHT

REWARD_BUMPINESS_PENALTY = 1  # Define penalty for bumpiness

class GameWrapper:
    def __init__(self):
        self.game = PentominoGame(headless=True)
        self.game.render_enabled = False

    def reset(self):
        self.game.reset_game()
        state_obj = PentominoGameState(self.game)
        # Initialize horizontal fill for comparison
        self.prev_horizontal_fill = self.calculate_horizontal_fill()
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

        # Calculate reward
        lines_cleared = self.game.lines_cleared - initial_lines
        reward = 0
        if lines_cleared > 0:
            reward += 100 * lines_cleared  # Strong reward for clearing lines

        # Penalty for holes
        reward -= 5 * self.count_holes()

        # Penalty for high stacks
        max_height = max(self.calculate_column_heights())
        reward -= 2 * max(0, max_height - GRID_HEIGHT // 2)

        # Small penalty per step to encourage faster play
        reward -= 1

        # Compute bumpiness
        heights = self.calculate_column_heights()
        bumpiness = sum(abs(heights[i] - heights[i-1]) for i in range(1, len(heights)))
        reward -= bumpiness * REWARD_BUMPINESS_PENALTY

        # Horizontal fill metric
        current_horizontal_fill = self.calculate_horizontal_fill()
        horizontal_diff = current_horizontal_fill - self.prev_horizontal_fill
        if horizontal_diff > 0:
            reward += horizontal_diff * 50  # Reward for improved horizontal fill
        self.prev_horizontal_fill = current_horizontal_fill

        # Near-complete lines setup
        near_complete = self.count_almost_complete_lines()
        reward += near_complete * 10  # Incentivize setting up lines

        done = self.game.game_over
        state_obj = PentominoGameState(self.game)
        next_state = state_obj.get_state_features().reshape(1, -1)
        return next_state, reward, done, {}

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

def train_agent(episodes=1000, print_every=10):
    env = GameWrapper()
    state_size = 10
    action_size = 5
    agent = LinearAgent(state_size, action_size)

    # Epsilon decay schedule parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    decay_rate = -np.log(EPSILON_END / EPSILON_START) / episodes

    scores = []
    lines_cleared = []
    epsilons = []
    reward_history = deque(maxlen=50)

    for ep in range(episodes):
        # Update epsilon using standard exponential decay
        agent.epsilon = max(EPSILON_END, EPSILON_START * np.exp(-decay_rate * ep))

        state = env.reset()
        done = False
        total_reward = 0
        start_score = env.game.score
        start_lines = env.game.lines_cleared

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        episode_score = env.game.score - start_score
        episode_lines = env.game.lines_cleared - start_lines

        scores.append(episode_score)
        lines_cleared.append(episode_lines)
        epsilons.append(agent.epsilon)
        reward_history.append(total_reward)

        if ep % print_every == 0:
            avg_reward = np.mean(reward_history)
            print(f"Ep {ep}/{episodes} | TotReward: {total_reward:.1f} | Score: {episode_score} | Lines: {episode_lines} | Epsilon: {agent.epsilon:.3f} | AvgReward: {avg_reward:.1f}")

    os.makedirs('models', exist_ok=True)
    agent.save('models/agent_online_q')
    print('Training complete. Model saved.')

    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(scores); plt.title('Score per Episode')
    plt.subplot(132)
    plt.plot(lines_cleared); plt.title('Lines Cleared per Episode')
    plt.subplot(133)
    plt.plot(epsilons); plt.title('Epsilon Decay')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_agent()