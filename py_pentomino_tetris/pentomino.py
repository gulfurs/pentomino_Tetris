"""
Pentomino shapes and related functions
Pentominoes are shapes made of 5 connected squares
"""

from constants import RED, GREEN, BLUE, DARK_BLUE, PURPLE, YELLOW

# Define pentomino shapes as a list of coordinates
# Each pentomino is defined by a list of (x, y) coordinates relative to a pivot point
PENTOMINOS = {
    'F': {
        'shape': [(0, 0), (1, 0), (0, 1), (-1, 1), (0, 2)],
        'color': RED
    },
    'I': {
        'shape': [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        'color': BLUE
    },
    'L': {
        'shape': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
        'color': DARK_BLUE
    },
    'P': {
        'shape': [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
        'color': PURPLE
    },
    'N': {
        'shape': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        'color': GREEN
    },
    'T': {
        'shape': [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        'color': YELLOW
    },
    'U': {
        'shape': [(0, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
        'color': RED
    },
    'V': {
        'shape': [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        'color': GREEN
    },
    'W': {
        'shape': [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        'color': DARK_BLUE
    },
    'X': {
        'shape': [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)],
        'color': PURPLE
    },
    'Y': {
        'shape': [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1)],
        'color': YELLOW
    },
    'Z': {
        'shape': [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
        'color': BLUE
    }
}


class Pentomino:
    def __init__(self, shape_name=None):
        """Initialize a new pentomino piece
        
        Args:
            shape_name (str, optional): The name of the pentomino shape. If None, a random shape is selected.
        """
        import random
        if shape_name is None:
            shape_name = random.choice(list(PENTOMINOS.keys()))
        
        self.shape_name = shape_name
        self.shape = PENTOMINOS[shape_name]['shape']
        self.color = PENTOMINOS[shape_name]['color']
        self.x = 0
        self.y = 0
        self.rotation = 0  # 0, 1, 2, or 3 representing 0, 90, 180, 270 degrees
    
    def get_coords(self):
        """Get the coordinates of the pentomino blocks in the current rotation
        
        Returns:
            List of (x, y) coordinates
        """
        rotated_shape = self._get_rotated_shape()
        return [(self.x + x, self.y + y) for x, y in rotated_shape]
    
    def _get_rotated_shape(self):
        """Get the shape rotated by the current rotation
        
        Returns:
            List of (x, y) coordinates
        """
        if self.rotation == 0:
            return self.shape
        elif self.rotation == 1:
            return [(-y, x) for x, y in self.shape]  # 90 degrees
        elif self.rotation == 2:
            return [(-x, -y) for x, y in self.shape]  # 180 degrees
        else:  # rotation == 3
            return [(y, -x) for x, y in self.shape]  # 270 degrees
    
    def rotate(self, clockwise=True):
        """Rotate the pentomino
        
        Args:
            clockwise (bool): True for clockwise, False for counter-clockwise
        """
        if clockwise:
            self.rotation = (self.rotation + 1) % 4
        else:
            self.rotation = (self.rotation - 1) % 4
    
    def move(self, dx, dy):
        """Move the pentomino by (dx, dy)
        
        Args:
            dx (int): Change in x
            dy (int): Change in y
        """
        self.x += dx
        self.y += dy