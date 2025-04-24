"""
Main Pentomino Tetris game file
"""

import pygame
import random
import sys
from constants import *
from pentomino import Pentomino

class PentominoGame:
    def __init__(self, headless=False):
        self.headless = headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Pentomino Tetris")
            self.clock = pygame.time.Clock()
        self.render_enabled = not self.headless  # Enable rendering if not headless
        
        # Load tile images
        self.tile_images = {
            RED: pygame.image.load('Tiles/Red_tile.png'),
            BLUE: pygame.image.load('Tiles/Blue_tile.png'),
            DARK_BLUE: pygame.image.load('Tiles/DarkBlue_tile.png'),
            GREEN: pygame.image.load('Tiles/Green_tile.png'),
            PURPLE: pygame.image.load('Tiles/Purple_tile.png'),
            YELLOW: pygame.image.load('Tiles/Yellow_tile.png')
        }
        self.grid_tile = pygame.image.load('Tiles/Grid_tile.png')
        
        # Resize images to match block size
        for color in self.tile_images:
            self.tile_images[color] = pygame.transform.scale(self.tile_images[color], (BLOCK_SIZE, BLOCK_SIZE))
        self.grid_tile = pygame.transform.scale(self.grid_tile, (BLOCK_SIZE, BLOCK_SIZE))
        
        # Game state
        self.reset_game()
        self.render_enabled = True  # Add this flag for enabling/disabling rendering
    
    def reset_game(self):
        """Reset the game state to start a new game"""
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Pentomino()
        self.current_piece.x = GRID_WIDTH // 2 - 1  # Start in the middle
        self.next_piece = Pentomino()
        self.score = 0
        self.level = 1
        self.game_over = False
        self.fall_speed = FALL_SPEED
        self.fall_time = 0
        self.lines_cleared = 0  # Add a counter for cleared lines
        self.action_history = []  # Track actions for better exploration
        self.episode_actions = 0  # Count actions in this episode
    
    def update(self, dt):
        """Update game state
        
        Args:
            dt: Time elapsed since last frame in milliseconds
        """
        if self.game_over:
            return
            
        self.fall_time += dt
        if self.fall_time >= (1000 / self.fall_speed):
            self.move_piece_down()
            self.fall_time = 0
    
    def move_piece_down(self):
        """Move the current piece down by one grid cell"""
        self.current_piece.move(0, 1)
        if self.check_collision():
            self.current_piece.move(0, -1)  # Move back up
            self.lock_piece()
            self.clear_lines()
            self.spawn_new_piece()
    
    def move_piece(self, dx):
        """Move the current piece horizontally
        
        Args:
            dx: Amount to move (-1 for left, 1 for right)
        """
        self.current_piece.move(dx, 0)
        if self.check_collision():
            self.current_piece.move(-dx, 0)  # Move back if collision
    
    def rotate_piece(self):
        """Rotate the current piece clockwise"""
        self.current_piece.rotate()
        if self.check_collision():
            # Try wall kicks
            kicks = [(1, 0), (-1, 0), (0, -1), (1, 1), (-1, 1)]
            for kick_x, kick_y in kicks:
                self.current_piece.move(kick_x, kick_y)
                if not self.check_collision():
                    return  # Found a valid position
                self.current_piece.move(-kick_x, -kick_y)
            
            # No valid position found, revert rotation
            self.current_piece.rotate(clockwise=False)
    
    def drop_piece(self):
        """Drop the current piece all the way down"""
        while not self.check_collision():
            self.current_piece.move(0, 1)
        self.current_piece.move(0, -1)  # Move back up from the collision position
        self.lock_piece()
        self.clear_lines()
        self.spawn_new_piece()
    
    def check_collision(self):
        """Check if the current piece collides with the grid boundaries or other blocks
        
        Returns:
            bool: True if collision detected, False otherwise
        """
        for x, y in self.current_piece.get_coords():
            if x < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
                return True
            if y >= 0 and self.grid[y][x] is not None:
                return True
        return False
    
    def lock_piece(self):
        """Lock the current piece onto the grid"""
        for x, y in self.current_piece.get_coords():
            if y >= 0:  # Only add to grid if within bounds
                self.grid[y][x] = self.current_piece.color
    
    def spawn_new_piece(self):
        """Spawn a new piece at the top of the grid"""
        self.current_piece = self.next_piece
        self.current_piece.x = GRID_WIDTH // 2 - 1
        self.current_piece.y = -2  # Start above visible grid to allow for tall pieces
        self.next_piece = Pentomino()
        
        # Check for game over
        if self.check_collision():
            self.game_over = True
    
    def clear_lines(self):
        """Clear completed lines and score points"""
        self.lines_cleared = 0
        
        for y in range(GRID_HEIGHT):
            if all(self.grid[y][x] is not None for x in range(GRID_WIDTH)):
                self.lines_cleared += 1
                # Move all lines above down
                for y2 in range(y, 0, -1):
                    self.grid[y2] = self.grid[y2-1].copy()
                self.grid[0] = [None] * GRID_WIDTH
        
        # Update score and level
        if self.lines_cleared > 0:
            self.score += self.lines_cleared ** 2 * 100 * self.level
            self.level = self.score // 1000 + 1
            self.fall_speed = FALL_SPEED + (self.level - 1) * FALL_SPEED_INCREMENT
    
    def render(self):
        """Render the game state to the screen"""
        # Skip rendering if disabled
        if self.headless or not self.render_enabled:
            return
            
        self.screen.fill(BLACK)
        
        # Draw grid background
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                self.screen.blit(self.grid_tile, (x * BLOCK_SIZE, y * BLOCK_SIZE))
        
        # Draw locked pieces
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] is not None:
                    self.screen.blit(self.tile_images[self.grid[y][x]], (x * BLOCK_SIZE, y * BLOCK_SIZE))
        
        # Draw current piece
        for x, y in self.current_piece.get_coords():
            if y >= 0:  # Only draw if within visible grid
                self.screen.blit(self.tile_images[self.current_piece.color], (x * BLOCK_SIZE, y * BLOCK_SIZE))
        
        # Draw sidebar
        sidebar_x = GRID_WIDTH * BLOCK_SIZE + 10
        
        # Draw score
        font = pygame.font.SysFont('Arial', 20)
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        level_text = font.render(f"Level: {self.level}", True, WHITE)
        self.screen.blit(score_text, (sidebar_x, 20))
        self.screen.blit(level_text, (sidebar_x, 50))
        
        # Draw next piece
        next_label = font.render("Next:", True, WHITE)
        self.screen.blit(next_label, (sidebar_x, 100))
        next_piece_preview = pygame.Surface((5 * BLOCK_SIZE, 5 * BLOCK_SIZE))
        next_piece_preview.fill(BLACK)
        
        # Center the next piece in the preview
        for x, y in self.next_piece.get_coords():
            next_piece_preview.blit(
                self.tile_images[self.next_piece.color], 
                ((x + 2) * BLOCK_SIZE, (y + 2) * BLOCK_SIZE)
            )
        self.screen.blit(next_piece_preview, (sidebar_x, 130))
        
        # Draw game over message
        if self.game_over:
            font = pygame.font.SysFont('Arial', 48)
            game_over_text = font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
    
    def handle_input(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if self.game_over:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.reset_game()
                continue
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.move_piece(-1)
                elif event.key == pygame.K_RIGHT:
                    self.move_piece(1)
                elif event.key == pygame.K_DOWN:
                    self.move_piece_down()
                elif event.key == pygame.K_UP:
                    self.rotate_piece()
                elif event.key == pygame.K_SPACE:
                    self.drop_piece()
    
    def run(self):
        """Run the main game loop"""
        while True:
            dt = self.clock.tick(FPS)
            self.handle_input()
            self.update(dt)
            self.render()


if __name__ == "__main__":
    game = PentominoGame()
    game.run()