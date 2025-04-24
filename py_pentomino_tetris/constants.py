"""
Constants for the Pentomino Tetris Game
"""

# Colors (RGB values)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 139)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (128, 128, 128)

# Game dimensions
BLOCK_SIZE = 30  # Size of each grid block in pixels
GRID_WIDTH = 15  # Width of the game grid (cells)
GRID_HEIGHT = 20  # Height of the game grid (cells)
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE + 200  # Additional space for score, next piece, etc.
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Game parameters
FPS = 60
FALL_SPEED = 1  # Base falling speed (blocks per second)
FALL_SPEED_INCREMENT = 0.1  # How much to increase speed as score increases