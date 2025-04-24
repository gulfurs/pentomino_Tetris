"""
Playback script for watching the trained RL agent play Pentomino Tetris
"""

import pygame
import time
import numpy as np
from rl_agent import LinearAgent, PentominoGameState
from game import PentominoGame
from constants import FPS, GRID_WIDTH, GRID_HEIGHT, BLOCK_SIZE, WHITE

def watch_agent_play(model_path="models/agent_final", episodes=5, delay=0.1):
    """Watch the trained agent play the game
    
    Args:
        model_path: Path to the saved model (without .npy extension)
        episodes: Number of episodes to play
        delay: Delay between actions in seconds (to slow down play)
    """
    # Initialize the game with rendering
    game = PentominoGame(headless=False)
    game.render_enabled = True
    
    # Set up the agent
    state_size = 10  # Updated to match enhanced features
    action_size = 5
    agent = LinearAgent(state_size, action_size)
    
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"Model file {model_path}.npy not found!")
        print("Make sure to train the agent first with train_agent.py")
        return
    
    # Display the weights the agent learned
    # agent.print_weights()
    
    # Start playing
    print(f"\nPlaying {episodes} episodes with the trained agent...")
    print("Close the window or press Esc to stop playback.")
    
    for episode in range(episodes):
        # Reset the game for a new episode
        game.reset_game()
        done = False
        total_score = 0
        steps = 0
        
        action_names = ["Left", "Right", "Rotate", "Soft Drop", "Hard Drop"]
        
        # Play until game over
        clock = pygame.time.Clock()
        
        while not done and steps < 5000:  # Limit steps to prevent infinite games
            # Get the current state
            state_obj = PentominoGameState(game)
            state = state_obj.get_state_features().reshape(1, -1)
            
            # Choose action
            action = agent.act(state)
            
            # Display action and state info
            state_features = state[0]
            pygame.display.set_caption(f"Episode {episode+1}/{episodes} | "
                                      f"Score: {game.score} | Action: {action_names[action]}")
            
            # Apply action
            if action == 0:
                game.move_piece(-1)
            elif action == 1:
                game.move_piece(1)
            elif action == 2:
                game.rotate_piece()
            elif action == 3:
                game.move_piece_down()
            elif action == 4:
                game.drop_piece()
            
            # Update and render the game
            dt = clock.tick(FPS)
            game.update(dt)
            game.render()
            
            # Add gameplay metrics to the display
            if steps % 5 == 0:  # Update every 5 steps to avoid flickering
                font = pygame.font.SysFont('Arial', 16)
                sidebar_x = GRID_WIDTH * BLOCK_SIZE + 10
                y_offset = 200
                
                # Show state information
                feature_names = ["Holes", "Bumpiness", "Max Height", "Complete Lines", 
                                "Near-complete Lines", "Line Opportunity", "Avg Height",
                                "Center Distance", "I-Piece", "Versatile Piece"]
                
                info_lines = [
                    "State Features:",
                    f"Holes: {state_features[0]:.2f}",
                    f"Bumpiness: {state_features[1]:.2f}",
                    f"Max Height: {state_features[2]:.2f}",
                    f"Lines: {state_features[3]:.2f}",
                    f"Action: {action_names[action]}",
                    f"Steps: {steps}"
                ]
                
                for line in info_lines:
                    text = font.render(line, True, WHITE)
                    game.screen.blit(text, (sidebar_x, y_offset))
                    y_offset += 20
                    
                pygame.display.flip()
            
            # Handle events (allow user to quit)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_SPACE:
                        # Speed toggle with spacebar
                        delay = 0 if delay > 0 else 0.1
            
            # Check if game is over
            done = game.game_over
            total_score = game.score
            steps += 1
            
            # Add delay to make it easier to watch
            time.sleep(delay)
        
        print(f"Episode {episode+1}: Score={total_score}, Steps={steps}")
        time.sleep(1)  # Pause between episodes
    
    print("Playback complete!")

if __name__ == "__main__":
    try:
        import sys
        model_path = "models/agent_final"
        episodes = 5
        delay = 0.1
        
        # Parse command line arguments if any
        if len(sys.argv) > 1:
            model_path = sys.argv[1]
        if len(sys.argv) > 2:
            episodes = int(sys.argv[2])
        if len(sys.argv) > 3:
            delay = float(sys.argv[3])
            
        pygame.init()
        watch_agent_play(model_path, episodes, delay)
    except KeyboardInterrupt:
        print("Playback stopped by user.")
    finally:
        pygame.quit()